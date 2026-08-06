"""Build a model-agnostic initial state from ECMWF open data, cached per init date.

Merges the retrieval halves of ECMWF's ``run_AIFS_v2.0.ipynb`` and
``run_AIFS_ENS_v2.0.ipynb``. The two models want *different* field sets, so retrieving
per-model would download each date twice; instead the **superset** is retrieved once and
cached one file per init date, and each model drops what it does not use
(``ModelSpec.drop_fields``). Getting the superset backwards is silent, so: ``PARAM_PL``
includes ``w`` (vertical velocity), used by ENS but not Single v2; both ``q_10`` and
``q_50`` are retrieved, and Single v2 drops both while ENS drops only ``q_10``; the
constant orography/mask fields are listed separately because they are published only on
the deterministic ``fc`` stream, which would matter if this were ever extended to
perturbed ensemble initial conditions.

The transformations below are not cosmetic -- they make open-data fields match what the
models were trained on: longitudes rolled to [0, 360); mean wave direction split into
cos/sin (a direction in degrees is discontinuous at 0/360 and cannot be interpolated or
predicted directly); soil variables renamed to their ERA5 names; snow depth and soil
moisture set to NaN over sea (the models were trained with them undefined there, and
anemoi re-imputes them); geopotential *height* converted to geopotential by multiplying
by g.
"""

from __future__ import annotations

import datetime
import logging
import pickle
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import earthkit.data as ekd
import numpy as np
import scipy.sparse as sp
from ecmwf.opendata import Client as OpendataClient

from .models import ModelSpec
from .paths import input_state_path
from .paths import lsm_path
from .regrid import LATLON_SHAPE
from .regrid import latlon_to_n320_matrix
from .regrid import to_n320

LOG = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SOURCES",
    "SOURCES",
    "build_input_state",
    "latest_date",
    "load_or_build_input_state",
    "select_for_model",
]

SOURCES = ("azure", "ecmwf", "aws", "google")

# Mirrors tried in order, best first. Measured 2026-08-06 with an identical 7.4 MB surface
# request straight to a temp file (no cache):
#
#   ecmwf    3.7 s   2039 kB/s   no retries
#   azure   15.5 s    487 kB/s   no retries
#   aws    370.8 s     20 kB/s   3x "503 Slow Down"
#   google      --         --    HTTP 400 Bad Request
#
# azure leads despite ecmwf's raw speed: the ECMWF portal caps total concurrent connections
# and degrades at peak times, whereas the cloud mirrors are replicas meant to absorb that
# load. google is excluded -- the URL layout ecmwf-opendata 0.3.29 builds for it does not
# resolve. End to end, a full initial state came down from azure in ~180 s.
DEFAULT_SOURCES = ("azure", "ecmwf", "aws")

PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "sd"]
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["vsw", "sot"]
PARAM_WAVE = ["wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "mwd", "cdww", "mwp", "swh"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
SOIL_LEVELS = [1, 2]

SOIL_RENAME = {"sot_1": "stl1", "sot_2": "stl2", "vsw_1": "swvl1", "vsw_2": "swvl2"}
LAND_ONLY = ["sd", "swvl1", "swvl2"]

G = 9.80665  # standard gravity, for geopotential height -> geopotential


def _as_sources(sources: str | Sequence[str]) -> tuple[str, ...]:
    return (sources,) if isinstance(sources, str) else tuple(sources)


def latest_date(sources: str | Sequence[str] = DEFAULT_SOURCES) -> datetime.datetime:
    """Most recent initialisation time available, trying each mirror in order."""
    failures = []
    for source in _as_sources(sources):
        try:
            return OpendataClient(source).latest()
        except Exception as exc:
            LOG.warning("Mirror %s could not report the latest date: %s", source, exc)
            failures.append(f"{source}: {exc}")
    raise RuntimeError("No open-data mirror responded:\n  " + "\n  ".join(failures))


def _from_source(sources: tuple[str, ...], **kwargs):
    """``ekd.from_source('ecmwf-open-data', ...)`` with per-request mirror fallback.

    Falling back per request rather than per run means one flaky batch does not throw away
    the batches already downloaded.
    """
    failures = []
    for source in sources:
        try:
            return ekd.from_source("ecmwf-open-data", source=source, **kwargs)
        except Exception as exc:
            LOG.warning("Mirror %s failed (%s); trying the next one", source, exc)
            failures.append(f"{source}: {exc}")
    raise RuntimeError(
        "Every mirror failed for this request:\n  "
        + "\n  ".join(failures)
        + "\nIf the date was resolved on one mirror, the fallbacks may not have published "
        "that run yet -- pass --date to pin an older initialisation."
    )


def _retrieve(
    param: list[str],
    dates: list[datetime.datetime],
    sources: tuple[str, ...],
    matrix: sp.csr_matrix,
    levelist: list[int] | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Download `param` at each date and regrid to N320."""
    levelist = levelist or []
    fields: dict[str, list[np.ndarray]] = defaultdict(list)

    for date in dates:
        LOG.info("Retrieving %s%s for %s", param, f" at {levelist}" if levelist else "", date)
        data = _from_source(sources, date=date, param=param, levelist=levelist, **kwargs)

        for f in data:
            values = f.to_numpy()
            if values.shape != LATLON_SHAPE:
                raise ValueError(f"Unexpected open-data grid {values.shape}, expected {LATLON_SHAPE}")
            # Open data runs -180..180; the models expect 0..360.
            values = np.roll(values, -values.shape[1] // 2, axis=1)
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
            fields[name].append(to_n320(values, matrix))

    return {name: np.stack(values) for name, values in fields.items()}


def _require(fields: dict, expected: list[str], what: str) -> None:
    missing = set(expected) - set(fields)
    if missing:
        raise RuntimeError(f"Open data is missing {what}: {sorted(missing)}")


def build_input_state(
    date: datetime.datetime | None = None,
    sources: str | Sequence[str] = DEFAULT_SOURCES,
    lsm: str | Path | None = None,
    regrid_directory: str | Path | None = None,
) -> dict:
    """Download and assemble the superset initial state for one date.

    Returns ``{"date": datetime, "fields": {name: (2, 542080) float32 array}}`` -- the two
    leading times are t-6h and t, in that order, as both models expect.
    """
    sources = _as_sources(sources)
    if date is None:
        date = latest_date(sources)
        LOG.info("Latest available initialisation: %s", date)

    LOG.info("Mirrors, in order of preference: %s", ", ".join(sources))
    matrix = latlon_to_n320_matrix(regrid_directory)

    # Both models are initialised from two analysis times, six hours apart.
    dates = [date - datetime.timedelta(hours=6), date]

    fields: dict[str, np.ndarray] = {}

    fields.update(_retrieve(PARAM_SFC, dates, sources, matrix, levtype="sfc"))
    _require(fields, PARAM_SFC, "surface fields")

    fields.update(_retrieve(PARAM_SFC_FC, dates, sources, matrix, levtype="sfc"))
    _require(fields, PARAM_SFC_FC, "constant surface fields")

    fields.update(_retrieve(PARAM_WAVE, dates, sources, matrix, stream="wave"))
    _require(fields, PARAM_WAVE, "wave fields")

    soil = _retrieve(PARAM_SOIL, dates, sources, matrix, levelist=SOIL_LEVELS)
    _require(soil, list(SOIL_RENAME), "soil fields")

    fields.update(_retrieve(PARAM_PL, dates, sources, matrix, levelist=LEVELS))
    _require(fields, [f"{p}_{lev}" for p in PARAM_PL for lev in LEVELS], "pressure-level fields")

    # --- transformations -------------------------------------------------------

    mwd_rad = np.deg2rad(fields.pop("mwd"))
    fields["cos_mwd"] = np.cos(mwd_rad)
    fields["sin_mwd"] = np.sin(mwd_rad)

    for open_data_name, era5_name in SOIL_RENAME.items():
        fields[era5_name] = soil[open_data_name]

    sea = np.equal(ekd.from_source("file", str(lsm or lsm_path()))[0].to_numpy(flatten=True), 0)
    n_points = fields["2t"].shape[1]
    if sea.shape[0] != n_points:
        raise ValueError(f"Land-sea mask has {sea.shape[0]} points but fields have {n_points}")
    for name in LAND_ONLY:
        fields[name][:, sea] = np.nan

    for level in LEVELS:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * G

    # The sparse matmul runs in float64; anemoi casts the input tensor to float32 anyway,
    # so downcast here and halve both memory and the size of the cached state.
    fields = {name: values.astype(np.float32) for name, values in fields.items()}

    LOG.info("Built superset input state: date=%s, %d fields", date, len(fields))
    return {"date": date, "fields": fields}


def load_or_build_input_state(
    date: datetime.datetime,
    sources: str | Sequence[str] = DEFAULT_SOURCES,
    cache: bool = True,
    **kwargs,
) -> dict:
    """Return the superset state for `date`, downloading it only if not already cached.

    The cache is shared by both models, so running Single v2 and then ENS for the same date
    downloads nothing the second time.
    """
    path = input_state_path(date)
    if path.exists():
        LOG.info("Using cached initial state %s", path)
        with path.open("rb") as f:
            return pickle.load(f)

    state = build_input_state(date=date, sources=sources, **kwargs)

    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary sibling and rename, so an interrupted run cannot leave a
        # truncated pickle that later looks like a valid cache hit.
        staged = path.with_suffix(".pkl.partial")
        with staged.open("wb") as f:
            pickle.dump(state, f)
        staged.rename(path)
        LOG.info("Cached initial state at %s", path)

    return state


def select_for_model(state: dict, spec: ModelSpec) -> dict:
    """Narrow the superset state to the fields `spec` expects.

    Returns a new dict; the cached superset is left untouched so the other model can still
    use it.
    """
    dropped = [name for name in spec.drop_fields if name in state["fields"]]
    fields = {k: v for k, v in state["fields"].items() if k not in spec.drop_fields}
    LOG.info(
        "%s input: %d fields (dropped %d: %s)",
        spec.pretty,
        len(fields),
        len(dropped),
        ", ".join(dropped) if dropped else "none",
    )
    return {"date": state["date"], "fields": fields}
