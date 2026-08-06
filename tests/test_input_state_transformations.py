"""The initial-condition transformations, which fail silently or not at all.

`build_input_state`'s transformations (listed in :mod:`aifs_mps.opendata`) are each a
*relabelling or rescaling of plausible numbers*. Get one wrong and nothing raises: the run
completes, the Zarr store is well-formed, and the map looks like weather -- just not
tomorrow's. A half-grid longitude error, a soil temperature crossed with a soil-moisture
field, or one pressure level left in metres instead of m^2/s^2 all reach both models
identically, because both draw from the same superset state.

Nothing here touches the network or the real grids: `ekd.from_source` is replaced (for the
downloads *and* the land-sea mask file), `LATLON_SHAPE` is shrunk to a 3x8 toy grid, and the
N320 regrid operator is replaced by the identity, so every retrieved value can be followed
to its exact place in the assembled state. Field values encode their own parameter, level,
time and grid position, which is what lets these tests tell "the right transformation" from
"the right transformation applied to the neighbouring field".
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest
import scipy.sparse as sp

from aifs_mps import opendata
from aifs_mps import regrid
from aifs_mps.opendata import LAND_ONLY
from aifs_mps.opendata import LEVELS
from aifs_mps.opendata import PARAM_PL
from aifs_mps.opendata import PARAM_SFC
from aifs_mps.opendata import PARAM_SFC_FC
from aifs_mps.opendata import PARAM_SOIL
from aifs_mps.opendata import PARAM_WAVE
from aifs_mps.opendata import SOIL_LEVELS
from aifs_mps.opendata import SOIL_RENAME
from aifs_mps.opendata import G
from aifs_mps.opendata import build_input_state

INIT = datetime.datetime(2026, 8, 6, 12)
EARLIER = INIT - datetime.timedelta(hours=6)

SMALL_SHAPE = (3, 8)  # (lat, lon); an even lon count, like the real 1440
HALF = SMALL_SHAPE[1] // 2
N_SMALL = SMALL_SHAPE[0] * SMALL_SHAPE[1]

# Every parameter the retrieval asks for, and every level slot it can ask for it at.
PARAMS = PARAM_SFC + PARAM_SFC_FC + PARAM_WAVE + PARAM_SOIL + PARAM_PL
LEVEL_SLOTS = [None, *SOIL_LEVELS, *LEVELS]


def encoded(param: str, level: int | None, date: datetime.datetime) -> np.ndarray:
    """A grid whose every value names the field, the time and the point it came from.

    The arithmetic keeps values small integers so they survive the float32 downcast
    exactly, which lets the assertions below use exact equality: any field that has been
    swapped with, overwritten by, or shifted relative to another shows up as a
    recognisably *wrong* number rather than a small numerical difference.
    """
    code = PARAMS.index(param) * len(LEVEL_SLOTS) + LEVEL_SLOTS.index(level)
    point = np.arange(N_SMALL, dtype=np.float64).reshape(SMALL_SHAPE)
    return (code * 100 + date.hour) * 100 + point


def rolled(grid: np.ndarray) -> np.ndarray:
    """The [-180, 180) -> [0, 360) roll, spelled out rather than reusing `np.roll`."""
    return np.concatenate([grid[:, HALF:], grid[:, :HALF]], axis=1)


def expected(param: str, level: int | None, date: datetime.datetime) -> np.ndarray:
    """What `encoded` must look like once it has been rolled and (identity-)regridded."""
    return rolled(encoded(param, level, date)).reshape(-1)


class FakeField:
    """Stands in for an earthkit field: values plus the two metadata keys used."""

    def __init__(self, param: str, level: int | None, values: np.ndarray):
        self.param = param
        self.level = level
        self.values = np.asarray(values, dtype=np.float64)

    def to_numpy(self, flatten: bool = False) -> np.ndarray:
        return self.values.reshape(-1).copy() if flatten else self.values.copy()

    def metadata(self, key: str):
        if key == "param":
            return self.param
        if key == "levelist":
            return self.level
        raise KeyError(key)


@pytest.fixture
def stub(monkeypatch):
    """Replace the network, the land-sea mask file, the grid size and the regrid operator.

    `values(param, level, date) -> (3, 8) array` decides what each retrieved field holds,
    so a test can feed real angles to `mwd` or real longitudes to everything. `mask` is the
    land-sea mask over the toy grid; it is all-land by default so tests that are not about
    sea masking see no NaNs.
    """

    def install(values=encoded, mask=None):
        monkeypatch.setattr(opendata, "LATLON_SHAPE", SMALL_SHAPE)
        monkeypatch.setattr(regrid, "LATLON_SHAPE", SMALL_SHAPE)
        # Identity: the regridding itself is covered by test_regrid.py, and an identity
        # operator keeps every source point individually addressable in the output.
        identity = sp.csr_matrix(np.eye(N_SMALL))
        monkeypatch.setattr(opendata, "latlon_to_n320_matrix", lambda directory=None: identity)

        land_sea = np.ones(N_SMALL) if mask is None else np.asarray(mask, dtype=np.float64)
        requests: list[dict] = []

        def from_source(name, *args, **kwargs):
            if name == "file":
                return [FakeField("lsm", None, land_sea)]
            assert name == "ecmwf-open-data", name
            requests.append(kwargs)
            levels = kwargs["levelist"] or [None]
            return [
                FakeField(param, level, values(param, level, kwargs["date"]))
                for param in kwargs["param"]
                for level in levels
            ]

        monkeypatch.setattr(opendata.ekd, "from_source", from_source)
        return requests

    return install


def build(stub, values=encoded, mask=None) -> dict:
    stub(values=values, mask=mask)
    return build_input_state(date=INIT, lsm="the stub never opens this")


# -- 1. the longitude roll -------------------------------------------------------


def geographic(param, level, date) -> np.ndarray:
    """Latitude index in the thousands, source longitude in [-180, 180) in the units."""
    lat = np.arange(SMALL_SHAPE[0], dtype=np.float64)[:, None] * 1000.0
    lon = np.linspace(-180.0, 180.0, SMALL_SHAPE[1], endpoint=False)[None, :]
    return lat + lon


def test_the_roll_puts_the_prime_meridian_in_column_zero(stub):
    """Open data starts at 180W, the models start at 0E. Rolling by the wrong amount --
    even by one column -- shifts the whole atmosphere sideways relative to the orography
    and the land-sea mask, which no plot of the result would reveal.

    Because each value carries its row as well as its longitude, this also pins the roll to
    `axis=1`: rolling latitudes instead would pass any longitude-only check.
    """
    fields = build(stub, values=geographic)["fields"]
    got = fields["2t"][1].reshape(SMALL_SHAPE)

    lat = np.arange(SMALL_SHAPE[0], dtype=np.float64)[:, None] * 1000.0
    # Column j must now hold the source point j*45 degrees east of Greenwich, which the
    # source labelled with that longitude wrapped into [-180, 180).
    east = np.linspace(0.0, 360.0, SMALL_SHAPE[1], endpoint=False)
    label = (east + 180.0) % 360.0 - 180.0

    np.testing.assert_array_equal(got, lat + label[None, :])
    assert (got[:, 0] == lat[:, 0]).all(), "column 0 must be longitude 0, not 180W"


def test_each_field_carries_its_own_values_through_the_roll(stub):
    """The roll happens field by field inside the retrieval loop; a field picking up its
    neighbour's array there would be invisible downstream."""
    fields = build(stub)["fields"]

    for name, param, level in [
        ("2t", "2t", None), ("msl", "msl", None), ("swh", "swh", None),
        ("sdor", "sdor", None), ("t_850", "t", 850), ("q_50", "q", 50), ("u_1000", "u", 1000),
    ]:
        np.testing.assert_array_equal(fields[name][1], expected(param, level, INIT), err_msg=name)


# -- 2. mean wave direction -> cos/sin -------------------------------------------


ANGLES = np.array([0.0, 1.0, 45.0, 90.0, 180.0, 270.0, 359.0, 359.9])


def wave_direction(param, level, date):
    if param == "mwd":
        return np.broadcast_to(ANGLES, SMALL_SHAPE).copy()
    return encoded(param, level, date)


def _mwd(fields):
    """The rolled angles, and the cos/sin pair the state should hold for them."""
    angles = rolled(np.broadcast_to(ANGLES, SMALL_SHAPE).copy())[0]
    return angles, fields["cos_mwd"][1].reshape(SMALL_SHAPE)[0], fields["sin_mwd"][1].reshape(SMALL_SHAPE)[0]


def test_wave_direction_is_replaced_by_the_cosine_and_sine_of_its_radians(stub):
    """Degrees fed straight to np.cos would still yield values in [-1, 1] -- a perfectly
    well-scaled field that means nothing. And leaving `mwd` in place would hand the model
    an input its checkpoint has no weights for."""
    fields = build(stub, values=wave_direction)["fields"]

    assert "mwd" not in fields, "the raw direction must not survive alongside cos/sin"
    angles, cos, sin = _mwd(fields)
    np.testing.assert_allclose(cos, np.cos(np.deg2rad(angles)), atol=1e-6)
    np.testing.assert_allclose(sin, np.sin(np.deg2rad(angles)), atol=1e-6)


def test_the_direction_survives_a_round_trip_through_cos_and_sin(stub):
    """cos/sin is only worth the two extra fields if the angle is recoverable from them."""
    fields = build(stub, values=wave_direction)["fields"]

    angles, cos, sin = _mwd(fields)
    np.testing.assert_allclose(np.rad2deg(np.arctan2(sin, cos)) % 360.0, angles % 360.0, atol=1e-3)


def test_directions_either_side_of_north_are_close_in_cos_sin_space(stub):
    """The whole point of the split: 1 deg and 359 deg are one degree apart on the compass
    but 358 apart as numbers, and anything that interpolates or regrids the raw degrees
    would place their midpoint due south."""
    fields = build(stub, values=wave_direction)["fields"]

    angles, cos, sin = _mwd(fields)
    near_zero = int(np.argmin(np.abs(angles - 1.0)))
    near_360 = int(np.argmin(np.abs(angles - 359.0)))
    assert abs(angles[near_zero] - angles[near_360]) > 350.0, "the raw degrees are far apart"

    separation = np.hypot(cos[near_zero] - cos[near_360], sin[near_zero] - sin[near_360])
    assert separation < 0.05, f"cos/sin should be continuous across north, got {separation}"


# -- 3. soil fields renamed to their ERA5 names ----------------------------------


def test_soil_fields_are_renamed_without_crossing_temperature_and_moisture(stub):
    """`sot` is a temperature and `vsw` a volumetric water content; a mapping that sent
    `sot_1` to `swvl1` would feed the model ~280 where it expects ~0.3, at the one place in
    the pipeline where the names stop matching the data."""
    mask = np.ones(N_SMALL)  # all land, so nothing is NaN'd out from under the comparison
    fields = build(stub, mask=mask)["fields"]

    for open_data_name, era5_name in SOIL_RENAME.items():
        param, level = open_data_name.split("_")
        np.testing.assert_array_equal(
            fields[era5_name], np.stack([expected(param, int(level), d) for d in (EARLIER, INIT)]),
            err_msg=f"{open_data_name} -> {era5_name}",
        )

    for open_data_name in SOIL_RENAME:
        assert open_data_name not in fields, f"{open_data_name} must not reach the model"
    assert SOIL_RENAME == {"sot_1": "stl1", "sot_2": "stl2", "vsw_1": "swvl1", "vsw_2": "swvl2"}


# -- 4. sea masking of the land-only fields --------------------------------------


SEA_POINTS = [0, 3, 7, 11, 12, 23]
MIXED_MASK = np.array([0.0 if i in SEA_POINTS else 1.0 for i in range(N_SMALL)])


def test_land_only_fields_are_blanked_exactly_over_sea(stub):
    """Snow depth and soil moisture were trained as undefined over sea and are re-imputed
    downstream. Masking the wrong points -- or treating the 0/1 mask as an index list --
    puts real values where the model expects none and vice versa."""
    fields = build(stub, mask=MIXED_MASK)["fields"]
    sea = MIXED_MASK == 0

    # Masking only the first of the three is the plausible bug, so all three are checked.
    sources = {"sd": ("sd", None), "swvl1": ("vsw", 1), "swvl2": ("vsw", 2)}
    assert set(sources) == set(LAND_ONLY), f"LAND_ONLY changed to {LAND_ONLY}"
    for name, (param, level) in sources.items():
        for t, date in enumerate((EARLIER, INIT)):
            values = fields[name][t]
            np.testing.assert_array_equal(
                np.isnan(values), sea, err_msg=f"{name} at t={t} is NaN in the wrong places"
            )
            np.testing.assert_array_equal(
                values[~sea], expected(param, level, date)[~sea].astype(np.float32),
                err_msg=f"{name} at t={t} lost data over land",
            )


def test_fields_that_are_defined_over_sea_are_left_alone(stub):
    """Wave height and 2 m temperature are defined everywhere; widening the mask to all
    fields would blank most of the globe's atmosphere before the first step."""
    fields = build(stub, mask=MIXED_MASK)["fields"]

    for name in ("2t", "swh", "stl1", "z_500"):
        assert not np.isnan(fields[name]).any(), f"{name} must not be masked"


# -- 5. geopotential height -> geopotential --------------------------------------


def test_every_pressure_level_is_converted_from_height_to_geopotential(stub):
    """One level left in metres is a factor-9.8 error on a single surface -- the forecast
    stays finite and the other thirteen levels look right."""
    fields = build(stub)["fields"]

    for level in LEVELS:
        assert f"gh_{level}" not in fields, f"gh_{level} must not reach the model"
        for t, date in enumerate((EARLIER, INIT)):
            np.testing.assert_allclose(
                fields[f"z_{level}"][t],
                (expected("gh", level, date) * G).astype(np.float32),
                rtol=1e-6,
                err_msg=f"z_{level} at t={t}",
            )

    assert not [name for name in fields if name.startswith("gh")]


def test_the_surface_geopotential_is_not_overwritten_by_a_level(stub):
    """`z` (orography) and `z_<level>` are different variables that differ by a suffix; the
    conversion writes into the same dict the surface field lives in."""
    fields = build(stub)["fields"]

    np.testing.assert_array_equal(fields["z"][1], expected("z", None, INIT))


# -- 6. dtype ---------------------------------------------------------------------


def test_every_field_is_float32(stub):
    """The sparse matmul runs in float64; a field left that way doubles a ~500 MB cached
    state and is silently downcast later anyway."""
    fields = build(stub)["fields"]

    wrong = {name: values.dtype for name, values in fields.items() if values.dtype != np.float32}
    assert wrong == {}


# -- 7. the two leading times ------------------------------------------------------


def test_the_two_leading_times_are_t_minus_six_then_t(stub):
    """Both checkpoints read index 0 as the older analysis. Reversed, the model is handed
    the six-hour tendency backwards and forecasts the recent past forwards -- a run that
    completes and verifies badly for no visible reason."""
    stub()
    state = build_input_state(date=INIT, lsm="the stub never opens this")
    fields = state["fields"]

    assert state["date"] == INIT
    assert fields["2t"].shape == (2, N_SMALL)
    np.testing.assert_array_equal(fields["2t"][0], expected("2t", None, EARLIER))
    np.testing.assert_array_equal(fields["2t"][1], expected("2t", None, INIT))


def test_both_analysis_times_are_requested_in_order_for_every_batch(stub):
    requests = stub()
    build_input_state(date=INIT, lsm="the stub never opens this")

    dates = [r["date"] for r in requests]
    assert dates, "nothing was requested"
    assert dates == [EARLIER, INIT] * (len(dates) // 2)
