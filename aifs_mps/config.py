"""Load and validate the Zarr output specification from YAML.

``config/default.yaml`` describes the store -- which fields, their units, which are
aggregated -- and this module turns it into the :class:`StoreSpec` that the layout and the
streaming writer are both built from. So adding a variable, changing its units, or asking
for a daily minimum is a one-line config edit, not a code change in three places.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "AGGREGATIONS",
    "DAILY_AGGREGATIONS",
    "Encoding",
    "OutputSpec",
    "StoreSpec",
    "load_store_spec",
    "native_only_spec",
]

NATIVE = "native"
DAILY_MEAN = "daily_mean"
DAILY_MIN = "daily_min"
DAILY_MAX = "daily_max"
DAILY_SUM = "daily_sum"

DAILY_AGGREGATIONS = (DAILY_MEAN, DAILY_MIN, DAILY_MAX, DAILY_SUM)
AGGREGATIONS = (NATIVE, *DAILY_AGGREGATIONS)

# Suffix appended to the output name when a variable produces more than one output.
_SUFFIX = {NATIVE: "", DAILY_MEAN: "_mean", DAILY_MIN: "_min", DAILY_MAX: "_max", DAILY_SUM: "_sum"}

# CF-style cell_methods, and the human-readable description written to each variable.
_CELL_METHOD = {
    DAILY_MEAN: "prediction_timedelta: mean",
    DAILY_MIN: "prediction_timedelta: minimum",
    DAILY_MAX: "prediction_timedelta: maximum",
    DAILY_SUM: "prediction_timedelta: sum",
}
_DESCRIPTION = {
    DAILY_MEAN: (
        "Daily mean of the four 6-hourly instantaneous values (06/12/18/24 UTC) in the "
        "UTC calendar day ending at time + prediction_timedelta_daily."
    ),
    DAILY_MIN: (
        "Minimum of the four 6-hourly instantaneous values (06/12/18/24 UTC) in the UTC "
        "calendar day ending at time + prediction_timedelta_daily. Note this is the "
        "minimum of four samples, not the true daily minimum."
    ),
    DAILY_MAX: (
        "Maximum of the four 6-hourly instantaneous values (06/12/18/24 UTC) in the UTC "
        "calendar day ending at time + prediction_timedelta_daily. Note this is the "
        "maximum of four samples, not the true daily maximum."
    ),
    DAILY_SUM: (
        "Accumulation over the UTC calendar day ending at time + "
        "prediction_timedelta_daily (sum of four 6-hourly accumulations)."
    ),
}

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.yaml"

# Fallback encoding, used for any key the config omits. This is the AIFS archive layout.
DEFAULT_CHUNKS = {
    "time": 1,
    "number": 1,
    "prediction_timedelta": 24,
    "prediction_timedelta_daily": 10,
    "lat": 90,
    "lon": 180,
}
DEFAULT_SHARDS = {
    "time": 1,
    "number": 1,
    "prediction_timedelta": 168,
    "prediction_timedelta_daily": 50,
    "lat": 720,
    "lon": 1440,
}
DEFAULT_COMPRESSOR = {"cname": "zstd", "clevel": 7, "shuffle": "bitshuffle"}
DEFAULT_KEEPBITS = 11


@dataclass(frozen=True)
class OutputSpec:
    """One stored variable: where it comes from and how it is reduced in time."""

    name: str

    source: str
    """Model field it is computed from. Several outputs may share one source."""

    aggregation: str
    units: str | None = None

    @property
    def is_native(self) -> bool:
        return self.aggregation == NATIVE

    @property
    def attrs(self) -> dict[str, str]:
        """Variable attributes, including the aggregation that produced it."""
        out: dict[str, str] = {"aggregation": self.aggregation}
        if self.units is not None:
            out["units"] = self.units
        if self.aggregation != NATIVE:
            out["cell_methods"] = _CELL_METHOD[self.aggregation]
            out["description"] = _DESCRIPTION[self.aggregation]
        if self.source != self.name:
            out["source_variable"] = self.source
        return out


@dataclass(frozen=True)
class Encoding:
    chunks: dict[str, int] = dataclass_field(default_factory=lambda: dict(DEFAULT_CHUNKS))
    shards: dict[str, int] = dataclass_field(default_factory=lambda: dict(DEFAULT_SHARDS))
    compressor: dict[str, Any] = dataclass_field(default_factory=lambda: dict(DEFAULT_COMPRESSOR))
    keepbits: int | None = DEFAULT_KEEPBITS
    keepbits_by_variable: dict[str, int | None] = dataclass_field(default_factory=dict)

    def keepbits_for(self, name: str) -> int | None:
        return self.keepbits_by_variable.get(name, self.keepbits)

    def validate(self) -> None:
        for dim, chunk in self.chunks.items():
            shard = self.shards.get(dim)
            if shard is None:
                raise ValueError(f"encoding.shards is missing dimension {dim!r}")
            if chunk <= 0 or shard <= 0:
                raise ValueError(f"encoding sizes must be positive; got {dim}: {chunk}/{shard}")
            if shard % chunk:
                raise ValueError(
                    f"encoding.shards[{dim}] = {shard} must be a positive multiple of "
                    f"encoding.chunks[{dim}] = {chunk}"
                )


@dataclass(frozen=True)
class StoreSpec:
    """The complete description of a forecast store."""

    outputs: tuple[OutputSpec, ...]
    encoding: Encoding = dataclass_field(default_factory=Encoding)
    path: Path | None = None

    @property
    def native(self) -> tuple[OutputSpec, ...]:
        return tuple(o for o in self.outputs if o.is_native)

    @property
    def daily(self) -> tuple[OutputSpec, ...]:
        return tuple(o for o in self.outputs if not o.is_native)

    @property
    def source_fields(self) -> tuple[str, ...]:
        """Model fields that must be retrieved, deduplicated and order-stable.

        Several outputs can share a source (2t feeding native, daily_min and daily_max);
        it must only be regridded once per step.
        """
        seen: dict[str, None] = {}
        for output in self.outputs:
            seen.setdefault(output.source, None)
        return tuple(seen)

    def without_daily(self) -> StoreSpec:
        """Same variables, everything at 6-hourly resolution (``--no-daily-aggregates``).

        Sources that only appeared as aggregates come back as native outputs under their
        source name.
        """
        outputs, seen = [], set()
        for source in self.source_fields:
            if source in seen:
                continue
            seen.add(source)
            units = next((o.units for o in self.outputs if o.source == source), None)
            outputs.append(OutputSpec(name=source, source=source, aggregation=NATIVE, units=units))
        return StoreSpec(tuple(outputs), self.encoding, self.path)

    def subset(self, names: list[str]) -> StoreSpec:
        """Restrict to the named *source* variables, keeping their aggregations."""
        unknown = sorted(set(names) - set(self.source_fields))
        if unknown:
            raise SystemExit(
                f"--save-fields: {', '.join(unknown)} not in {self.path or 'the config'}. "
                f"Available: {', '.join(self.source_fields)}"
            )
        keep = set(names)
        return StoreSpec(
            tuple(o for o in self.outputs if o.source in keep), self.encoding, self.path
        )


def _parse_encoding(raw: dict[str, Any] | None) -> Encoding:
    raw = raw or {}
    unknown = set(raw) - {"chunks", "shards", "compressor", "keepbits", "keepbits_by_variable"}
    if unknown:
        raise ValueError(f"Unknown encoding keys: {', '.join(sorted(unknown))}")

    encoding = Encoding(
        chunks={**DEFAULT_CHUNKS, **(raw.get("chunks") or {})},
        shards={**DEFAULT_SHARDS, **(raw.get("shards") or {})},
        compressor={**DEFAULT_COMPRESSOR, **(raw.get("compressor") or {})},
        # `keepbits: null` is meaningful (store exact float32), so distinguish it from absent.
        keepbits=raw.get("keepbits", DEFAULT_KEEPBITS),
        keepbits_by_variable=dict(raw.get("keepbits_by_variable") or {}),
    )
    encoding.validate()
    return encoding


def _parse_variables(raw: dict[str, Any]) -> tuple[OutputSpec, ...]:
    if not raw:
        raise ValueError("config must define at least one variable under `variables`")

    outputs: list[OutputSpec] = []
    for name, entry in raw.items():
        entry = entry or {}
        if not isinstance(entry, dict):
            raise ValueError(f"variables.{name} must be a mapping, got {type(entry).__name__}")
        unknown = set(entry) - {"units", "aggregations"}
        if unknown:
            raise ValueError(f"variables.{name}: unknown keys {', '.join(sorted(unknown))}")

        aggregations = entry.get("aggregations") or [NATIVE]
        if isinstance(aggregations, str):
            aggregations = [aggregations]
        bad = [a for a in aggregations if a not in AGGREGATIONS]
        if bad:
            raise ValueError(
                f"variables.{name}: unknown aggregation(s) {', '.join(bad)}; "
                f"valid: {', '.join(AGGREGATIONS)}"
            )
        if len(set(aggregations)) != len(aggregations):
            raise ValueError(f"variables.{name}: duplicate aggregations {aggregations}")

        # A single aggregation keeps the bare variable name, so a store built from the
        # default config is identical to what the reference scripts produced.
        suffixed = len(aggregations) > 1
        for aggregation in aggregations:
            outputs.append(
                OutputSpec(
                    name=name + (_SUFFIX[aggregation] if suffixed else ""),
                    source=name,
                    aggregation=aggregation,
                    units=entry.get("units"),
                )
            )

    names = [o.name for o in outputs]
    clashes = sorted({n for n in names if names.count(n) > 1})
    if clashes:
        raise ValueError(f"Config produces duplicate output names: {', '.join(clashes)}")
    return tuple(outputs)


def load_store_spec(path: str | Path | None = None) -> StoreSpec:
    """Read a store specification from YAML."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")

    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    unknown = set(raw) - {"variables", "encoding"}
    if unknown:
        raise SystemExit(f"{path}: unknown top-level keys {', '.join(sorted(unknown))}")

    try:
        spec = StoreSpec(
            outputs=_parse_variables(raw.get("variables") or {}),
            encoding=_parse_encoding(raw.get("encoding")),
            path=path,
        )
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from None
    return spec


def native_only_spec(fields: list[str], encoding: Encoding | None = None) -> StoreSpec:
    """Store every given field 6-hourly with no units (``--save-fields all``)."""
    return StoreSpec(
        tuple(OutputSpec(name=f, source=f, aggregation=NATIVE) for f in fields),
        encoding or Encoding(),
    )
