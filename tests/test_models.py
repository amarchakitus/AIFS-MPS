"""The model registry and the shared initial-condition cache.

The point of the shared cache is that one download serves both models. That only works if
`select_for_model` narrows the superset correctly: too few fields and the model fails
loudly, too many and it may fail *quietly* by feeding a variable the checkpoint does not
expect. These tests pin the field arithmetic so a future edit to the registry cannot
silently change what either model receives.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from aifs_mps.models import ENS
from aifs_mps.models import MODELS
from aifs_mps.models import SINGLE
from aifs_mps.models import check_runtime
from aifs_mps.models import get_model
from aifs_mps.opendata import LEVELS
from aifs_mps.opendata import select_for_model

# The superset the retrieval builds: what both models draw from.
SUPERSET_FIELDS = (
    ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "sd", "lsm", "z", "slor", "sdor"]
    + ["wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "cdww", "mwp", "swh"]
    + ["cos_mwd", "sin_mwd", "stl1", "stl2", "swvl1", "swvl2"]
    + [f"{p}_{lev}" for p in ("z", "t", "u", "v", "w", "q") for lev in LEVELS]
)


@pytest.fixture
def superset():
    rng = np.random.default_rng(0)
    return {
        "date": datetime.datetime(2026, 8, 6, 6),
        "fields": {name: rng.normal(size=(2, 8)).astype(np.float32) for name in SUPERSET_FIELDS},
    }


def test_registry_is_keyed_by_name():
    assert MODELS == {"single": SINGLE, "ens": ENS}
    assert get_model("ens") is ENS


def test_unknown_model_is_rejected():
    with pytest.raises(SystemExit, match="Unknown model"):
        get_model("nope")


def test_the_two_models_need_different_anemoi_versions():
    """If these ever converge, the two-runtime layout can be collapsed -- and should be."""
    assert SINGLE.anemoi_models != ENS.anemoi_models


def test_single_drops_vertical_velocity_and_upper_humidity(superset):
    selected = select_for_model(superset, SINGLE)["fields"]
    assert not [f for f in selected if f.startswith("w_")], "Single v2 does not use w"
    assert "q_50" not in selected and "q_10" not in selected
    assert "q_100" in selected, "humidity at and below 100 hPa is still a prognostic"


def test_ens_keeps_w_and_q50(superset):
    selected = select_for_model(superset, ENS)["fields"]
    assert len([f for f in selected if f.startswith("w_")]) == len(LEVELS)
    assert "q_50" in selected, "ENS keeps q_50 -- this is the classic copy-paste bug"
    assert "q_10" not in selected


def test_ens_receives_strictly_more_than_single(superset):
    single = set(select_for_model(superset, SINGLE)["fields"])
    ens = set(select_for_model(superset, ENS)["fields"])
    assert single < ens, "Single's field set should be a strict subset of ENS's"


def test_selection_does_not_mutate_the_shared_cache(superset):
    """Both models read the same cached object; neither may modify it for the other."""
    before = set(superset["fields"])
    select_for_model(superset, SINGLE)
    select_for_model(superset, ENS)
    assert set(superset["fields"]) == before


def test_selection_shares_arrays_rather_than_copying(superset):
    """A superset state is ~500 MB; selection must not duplicate it."""
    selected = select_for_model(superset, ENS)
    assert selected["fields"]["2t"] is superset["fields"]["2t"]
    assert selected["date"] == superset["date"]


def test_dropping_an_absent_field_is_not_an_error(superset):
    """Caches written by an older build may already lack a dropped field."""
    del superset["fields"]["q_10"]
    selected = select_for_model(superset, ENS)["fields"]
    assert "q_10" not in selected


# -- the runtime guard -----------------------------------------------------------
#
# The two checkpoints need incompatible anemoi-models versions, and loading one into the
# wrong environment dies with a ModuleNotFoundError raised from inside the unpickler --
# after the checkpoint has been read off disk. `check_runtime` is what turns that into an
# up-front message, so both halves matter: it must fire on a mismatch *and* it must tell
# the user which command to run instead. A guard that only says "wrong version" leaves
# them staring at two runtimes with no hint which one this model wants.


@pytest.fixture
def installed_anemoi(monkeypatch):
    def install(version: str):
        import anemoi.models

        monkeypatch.setattr(anemoi.models, "__version__", version)

    return install


@pytest.mark.parametrize("spec", [SINGLE, ENS], ids=lambda s: s.name)
def test_a_wrong_anemoi_version_aborts_and_names_the_command_to_use(spec, installed_anemoi):
    installed_anemoi("0.0.1-not-this-one")

    with pytest.raises(SystemExit) as excinfo:
        check_runtime(spec)

    message = str(excinfo.value)
    assert spec.anemoi_models in message, "the expected version must be stated"
    assert "0.0.1-not-this-one" in message, "so must the one actually installed"
    assert f"./aifs {spec.name}" in message
    assert f"runtimes/{spec.runtime}" in message


@pytest.mark.parametrize("spec", [SINGLE, ENS], ids=lambda s: s.name)
def test_the_matching_runtime_is_allowed_through(spec, installed_anemoi):
    """Over-strict matching would make both models unrunnable."""
    installed_anemoi(spec.anemoi_models)
    check_runtime(spec)
