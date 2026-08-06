"""Where the repo keeps its large, non-code assets.

Every location has an environment-variable override so the package works from a checkout,
an installed wheel, or a scratch directory on another machine.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "forecast_dir",
    "input_state_dir",
    "input_state_path",
    "lsm_path",
    "regrid_dir",
    "weights_dir",
]

REPO_ROOT = Path(__file__).resolve().parent.parent


def _dir(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, default))


def weights_dir() -> Path:
    """Model checkpoints (``AIFS_MPS_WEIGHTS_DIR``)."""
    return _dir("AIFS_MPS_WEIGHTS_DIR", REPO_ROOT / "weights")


def support_dir() -> Path:
    """Static inputs shipped with the repo (``AIFS_MPS_SUPPORT_DIR``)."""
    return _dir("AIFS_MPS_SUPPORT_DIR", REPO_ROOT / "support")


def regrid_dir() -> Path:
    """Interpolation matrices (``AIFS_MPS_REGRID_DIR``)."""
    return _dir("AIFS_MPS_REGRID_DIR", support_dir() / "regrid")


def lsm_path() -> Path:
    """N320 land-sea mask GRIB (``AIFS_MPS_LSM``)."""
    return Path(os.environ.get("AIFS_MPS_LSM", support_dir() / "lsm.grib"))


def input_state_dir() -> Path:
    """Shared initial-condition cache (``AIFS_MPS_IC_DIR``).

    Deliberately *not* per-model: one file per init date holds the superset of fields both
    models need, so a date is downloaded once and reused by either.
    """
    return _dir("AIFS_MPS_IC_DIR", REPO_ROOT / "input_states")


def input_state_path(date) -> Path:
    """Cache path for one initialisation date."""
    return input_state_dir() / f"input_state_{date.strftime('%Y%m%dT%H')}.pkl"


def forecast_dir(model: str | None = None) -> Path:
    """Output Zarr stores (``AIFS_MPS_FORECAST_DIR``).

    Per-model subdirectory: unlike the initial conditions, which are deliberately shared,
    the two models produce *different* forecasts for the same init date and would otherwise
    collide on ``init_<date>.zarr``.
    """
    root = _dir("AIFS_MPS_FORECAST_DIR", REPO_ROOT / "forecasts")
    return root / model if model else root
