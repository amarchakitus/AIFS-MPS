"""The model registry -- the one place AIFS Single v2 and AIFS-ENS v2 differ.

Everything else is written against :class:`ModelSpec`, so a third variant should be a new
entry here rather than a branch in the code.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MODELS", "ModelSpec", "get_model"]


@dataclass(frozen=True)
class ModelSpec:
    """Everything that distinguishes one AIFS variant from another."""

    name: str
    checkpoint: str
    """Checkpoint filename, also the name it is published under on Hugging Face."""

    hf_repo: str
    """Hugging Face repo holding the checkpoint. `aifs_mps.weights` turns this into a URL;
    the registry deliberately records which model, not how to fetch it."""

    runtime: str
    """Directory under runtimes/ holding this model's pinned anemoi-models version."""

    anemoi_models: str
    """Expected anemoi-models version, asserted at startup so a mismatched environment
    fails immediately with a clear message rather than deep inside unpickling."""

    ensemble: bool
    default_lead_time: int
    default_members: int

    drop_fields: tuple[str, ...] = ()
    """Variables present in the shared superset initial state that this model must not
    receive. Everything else is passed through."""

    description: str = ""

    @property
    def pretty(self) -> str:
        return f"{self.name} ({'ensemble' if self.ensemble else 'deterministic'})"


SINGLE = ModelSpec(
    name="single",
    checkpoint="aifs-single-mse-2.0.ckpt",
    hf_repo="ecmwf/aifs-single-2.0",
    runtime="single",
    anemoi_models="0.9.3",
    ensemble=False,
    default_lead_time=360,
    default_members=1,
    # Single v2 uses neither vertical velocity nor specific humidity above 100 hPa.
    drop_fields=(
        "q_10",
        "q_50",
        *(
            f"w_{lev}"
            for lev in (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10)
        ),
    ),
    description="AIFS Single v2, deterministic",
)

ENS = ModelSpec(
    name="ens",
    checkpoint="aifs-ens-crps-2.0.ckpt",
    hf_repo="ecmwf/aifs-ens-2.0",
    runtime="ens",
    anemoi_models="0.11.2",
    ensemble=True,
    default_lead_time=360,
    default_members=10,
    # ENS keeps q_50 and uses w; only the 10 hPa humidity level is unused.
    drop_fields=("q_10",),
    description="AIFS-ENS v2, stochastic ensemble",
)

MODELS: dict[str, ModelSpec] = {spec.name: spec for spec in (SINGLE, ENS)}


def get_model(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise SystemExit(
            f"Unknown model {name!r}; choose from {', '.join(sorted(MODELS))}"
        ) from None


def check_runtime(spec: ModelSpec) -> None:
    """Fail fast if this interpreter has the wrong anemoi-models for `spec`.

    Loading a checkpoint into the wrong version dies with a confusing ``ModuleNotFoundError``
    from deep inside the unpickler, so we check up front and say what to run instead.
    """
    import anemoi.models

    installed = anemoi.models.__version__
    if installed != spec.anemoi_models:
        raise SystemExit(
            f"{spec.pretty} needs anemoi-models {spec.anemoi_models}, but this environment "
            f"has {installed}.\nRun it through its own runtime:\n"
            f"    ./aifs {spec.name} ...\n"
            f"or:  uv run --project runtimes/{spec.runtime} aifs-{spec.name} ..."
        )
