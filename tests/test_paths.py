"""Where the large assets live, and who is allowed to share a directory with whom.

Every location has an environment override so the package can run from a checkout, a wheel,
or a scratch disk -- an override read once at import time instead of per call would silently
ignore the user's setting. And the sharing rules are deliberate and asymmetric: initial
conditions are shared between the two models on purpose (one download serves both), while
forecasts must be kept apart, because both models write ``init_<date>.zarr`` and the second
run used to overwrite the first.
"""

from __future__ import annotations

import datetime

import pytest

from aifs_mps import paths

ENV_VARS = [
    "AIFS_MPS_WEIGHTS_DIR",
    "AIFS_MPS_SUPPORT_DIR",
    "AIFS_MPS_REGRID_DIR",
    "AIFS_MPS_LSM",
    "AIFS_MPS_IC_DIR",
    "AIFS_MPS_FORECAST_DIR",
]


@pytest.fixture(autouse=True)
def pristine_environment(monkeypatch):
    """A developer machine may well have some of these set; start from none of them."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("env_var", "accessor"),
    [
        ("AIFS_MPS_WEIGHTS_DIR", paths.weights_dir),
        ("AIFS_MPS_SUPPORT_DIR", paths.support_dir),
        ("AIFS_MPS_REGRID_DIR", paths.regrid_dir),
        ("AIFS_MPS_LSM", paths.lsm_path),
        ("AIFS_MPS_IC_DIR", paths.input_state_dir),
        ("AIFS_MPS_FORECAST_DIR", paths.forecast_dir),
    ],
)
def test_every_location_honours_its_environment_override(monkeypatch, tmp_path, env_var, accessor):
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv(env_var, str(elsewhere))
    assert accessor() == elsewhere


def test_support_dir_override_carries_the_files_derived_from_it(monkeypatch, tmp_path):
    """The matrices and the land-sea mask are shipped *inside* support/, so relocating
    support/ has to move them too rather than leaving them pointing at the checkout."""
    monkeypatch.setenv("AIFS_MPS_SUPPORT_DIR", str(tmp_path))
    assert paths.regrid_dir() == tmp_path / "regrid"
    assert paths.lsm_path() == tmp_path / "lsm.grib"


def test_the_two_models_do_not_share_a_forecast_directory(monkeypatch, tmp_path):
    """Both models name their store ``init_<date>.zarr``. When these two paths were equal,
    running ens after single silently overwrote the deterministic forecast."""
    monkeypatch.setenv("AIFS_MPS_FORECAST_DIR", str(tmp_path))
    single, ens = paths.forecast_dir("single"), paths.forecast_dir("ens")

    assert single != ens
    assert single.parent == ens.parent == paths.forecast_dir()


def test_initial_conditions_are_shared_across_models_but_split_by_run(monkeypatch, tmp_path):
    """One file per init date, model-agnostic -- but 00Z and 12Z of the same day are
    different initial states and must not collide on one filename."""
    monkeypatch.setenv("AIFS_MPS_IC_DIR", str(tmp_path))
    midnight = paths.input_state_path(datetime.datetime(2026, 8, 6, 0))
    midday = paths.input_state_path(datetime.datetime(2026, 8, 6, 12))

    assert midnight.parent == midday.parent == tmp_path
    assert midnight != midday
    assert "single" not in midnight.name and "ens" not in midnight.name
