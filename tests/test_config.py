"""Parsing and validating the YAML store specification.

The config now decides what the store contains, so a silently-misread config produces a
plausible but wrong archive. These tests pin the naming rule, the aggregation set, and the
validation that should reject a bad file loudly at startup rather than mid-forecast.
"""

from __future__ import annotations

import textwrap

import pytest

from aifs_mps.config import AGGREGATIONS
from aifs_mps.config import DEFAULT_CONFIG
from aifs_mps.config import Encoding
from aifs_mps.config import load_store_spec
from aifs_mps.config import native_only_spec


def write(tmp_path, body: str):
    path = tmp_path / "spec.yaml"
    path.write_text(textwrap.dedent(body))
    return path


# -- the shipped default ---------------------------------------------------------


def test_default_config_loads():
    spec = load_store_spec()
    assert spec.path == DEFAULT_CONFIG
    assert len(spec.outputs) == 25
    assert len(spec.native) == 14
    assert len(spec.daily) == 11


def test_default_config_preserves_the_archive_variable_names():
    """Every default variable has one aggregation, so all names stay bare -- a store built
    from the shipped config must remain readable by existing archive tooling."""
    spec = load_store_spec()
    names = {o.name for o in spec.outputs}
    assert {"2t", "msl", "tcw", "z_500", "tp"} <= names
    assert not [n for n in names if n.endswith(("_mean", "_min", "_max", "_sum"))]
    for output in spec.outputs:
        assert output.name == output.source


def test_default_config_units_are_present():
    spec = load_store_spec()
    units = {o.name: o.units for o in spec.outputs}
    assert units["2t"] == "K"
    assert units["tp"] == "m"
    assert units["z_500"] == "m**2 s**-2"
    assert all(o.units for o in spec.outputs), "every configured variable should carry units"


# -- naming rule -----------------------------------------------------------------


def test_single_aggregation_keeps_the_bare_name(tmp_path):
    spec = load_store_spec(write(tmp_path, """
        variables:
          tcw: {units: kg m**-2, aggregations: [daily_mean]}
    """))
    assert [o.name for o in spec.outputs] == ["tcw"]


def test_multiple_aggregations_are_suffixed(tmp_path):
    spec = load_store_spec(write(tmp_path, """
        variables:
          2t: {units: K, aggregations: [native, daily_min, daily_max]}
    """))
    assert [o.name for o in spec.outputs] == ["2t", "2t_min", "2t_max"]
    assert all(o.source == "2t" for o in spec.outputs)
    # one source, so the field is retrieved and regridded once
    assert spec.source_fields == ("2t",)


def test_aggregation_defaults_to_native(tmp_path):
    spec = load_store_spec(write(tmp_path, "variables:\n  2t: {units: K}\n"))
    assert spec.outputs[0].aggregation == "native"


# -- attributes ------------------------------------------------------------------


def test_every_output_records_its_aggregation(tmp_path):
    spec = load_store_spec(write(tmp_path, """
        variables:
          2t: {units: K, aggregations: [native, daily_min]}
          tp: {units: m, aggregations: [daily_sum]}
    """))
    attrs = {o.name: o.attrs for o in spec.outputs}
    assert attrs["2t"]["aggregation"] == "native"
    assert attrs["2t_min"]["aggregation"] == "daily_min"
    assert attrs["tp"]["aggregation"] == "daily_sum"
    assert attrs["2t_min"]["cell_methods"] == "prediction_timedelta: minimum"
    assert attrs["2t_min"]["source_variable"] == "2t"
    assert "cell_methods" not in attrs["2t"], "native output is not a cell method"


# -- validation ------------------------------------------------------------------


def test_unknown_aggregation_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="unknown aggregation"):
        load_store_spec(write(tmp_path, "variables:\n  2t: {aggregations: [daily_median]}\n"))


def test_duplicate_aggregation_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="duplicate aggregations"):
        load_store_spec(write(tmp_path, "variables:\n  2t: {aggregations: [native, native]}\n"))


def test_unknown_variable_key_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="unknown keys"):
        load_store_spec(write(tmp_path, "variables:\n  2t: {unit: K}\n"))


def test_unknown_top_level_key_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="unknown top-level"):
        load_store_spec(write(tmp_path, "variables:\n  2t: {}\nvariabels: {}\n"))


def test_empty_config_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="at least one variable"):
        load_store_spec(write(tmp_path, "variables: {}\n"))


def test_missing_config_is_reported(tmp_path):
    with pytest.raises(SystemExit, match="Config not found"):
        load_store_spec(tmp_path / "nope.yaml")


def test_shard_must_be_a_multiple_of_chunk(tmp_path):
    with pytest.raises(SystemExit, match="must be a positive multiple"):
        load_store_spec(write(tmp_path, """
            variables:
              2t: {}
            encoding:
              chunks: {prediction_timedelta: 24}
              shards: {prediction_timedelta: 30}
        """))


# -- encoding defaults -----------------------------------------------------------


def test_encoding_falls_back_to_code_defaults(tmp_path):
    """Config need not mention encoding at all."""
    spec = load_store_spec(write(tmp_path, "variables:\n  2t: {}\n"))
    assert spec.encoding.chunks["prediction_timedelta"] == 24
    assert spec.encoding.shards["lon"] == 1440
    assert spec.encoding.keepbits == 11
    assert spec.encoding.compressor["cname"] == "zstd"


def test_partial_encoding_overrides_only_what_it_names(tmp_path):
    spec = load_store_spec(write(tmp_path, """
        variables:
          2t: {}
        encoding:
          keepbits: 7
    """))
    assert spec.encoding.keepbits == 7
    assert spec.encoding.chunks["lat"] == 90, "unspecified keys keep their defaults"


def test_null_keepbits_disables_bitround(tmp_path):
    """`keepbits: null` means exact float32 and must not be confused with 'absent'."""
    spec = load_store_spec(write(tmp_path, "variables:\n  2t: {}\nencoding:\n  keepbits: null\n"))
    assert spec.encoding.keepbits is None


def test_per_variable_keepbits_override(tmp_path):
    spec = load_store_spec(write(tmp_path, """
        variables:
          2t: {}
          tp: {}
        encoding:
          keepbits: 11
          keepbits_by_variable: {tp: 13}
    """))
    assert spec.encoding.keepbits_for("tp") == 13
    assert spec.encoding.keepbits_for("2t") == 11


# -- spec transforms used by the CLI ---------------------------------------------


def test_without_daily_flattens_everything_to_native(tmp_path):
    spec = load_store_spec(write(tmp_path, """
        variables:
          2t:  {units: K, aggregations: [native, daily_max]}
          tcw: {units: kg m**-2, aggregations: [daily_mean]}
    """)).without_daily()
    assert [(o.name, o.aggregation) for o in spec.outputs] == [("2t", "native"), ("tcw", "native")]
    assert spec.outputs[1].units == "kg m**-2", "units survive the flattening"


def test_subset_keeps_all_aggregations_of_the_kept_variable(tmp_path):
    spec = load_store_spec(write(tmp_path, """
        variables:
          2t:  {aggregations: [native, daily_max]}
          tcw: {aggregations: [daily_mean]}
    """)).subset(["2t"])
    assert [o.name for o in spec.outputs] == ["2t", "2t_max"]


def test_subset_rejects_an_unknown_name(tmp_path):
    spec = load_store_spec(write(tmp_path, "variables:\n  2t: {}\n"))
    with pytest.raises(SystemExit, match="not in"):
        spec.subset(["nope"])


def test_native_only_spec_is_all_native():
    spec = native_only_spec(["a", "b"])
    assert [o.aggregation for o in spec.outputs] == ["native", "native"]
    assert isinstance(spec.encoding, Encoding)


def test_all_aggregations_are_reachable_from_config(tmp_path):
    """Every name in AGGREGATIONS must actually parse -- catches a constant added to the
    tuple but not wired into the suffix/cell_methods tables."""
    body = "variables:\n" + "".join(
        f"  v{i}: {{aggregations: [{a}]}}\n" for i, a in enumerate(AGGREGATIONS)
    )
    spec = load_store_spec(write(tmp_path, body))
    assert {o.aggregation for o in spec.outputs} == set(AGGREGATIONS)
    for output in spec.outputs:
        assert output.attrs["aggregation"] == output.aggregation
