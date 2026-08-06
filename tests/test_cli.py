"""The two parts of the CLI that are wrong before anything runs, not while it runs.

`preresolve_num_chunks` exists only because of an ordering constraint: anemoi reads
`ANEMOI_INFERENCE_NUM_CHUNKS` into module-level constants at *import* time, so the value
has to be in `os.environ` before `anemoi.models` is imported -- which is before argparse
has seen the real command line. If it silently fails to export, the run still succeeds; it
just quietly uses 1 chunk and peaks at ~88 GB instead of ~20 GB. Nothing downstream
notices, so it is asserted here.

`build_parser` is the only place the two models' interfaces differ. `--members` and
`--seed-offset` are meaningless for the deterministic model, and an argparse mistake there
would make `aifs single --members 10` accept the flag and produce one member anyway.

Nothing here runs a forecast: `build_parser` is called directly with a `ModelSpec`.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from aifs_mps.cli import DEFAULT_NUM_CHUNKS
from aifs_mps.cli import build_parser
from aifs_mps.cli import preresolve_num_chunks
from aifs_mps.models import ENS
from aifs_mps.models import SINGLE
from aifs_mps.models import get_model

NUM_CHUNKS_ENV = "ANEMOI_INFERENCE_NUM_CHUNKS"


@pytest.fixture(autouse=True)
def stale_num_chunks(monkeypatch):
    """Start every test with a wrong value already exported, and restore it afterwards.

    A leftover value from a previous invocation must be overwritten, not deferred to --
    `setdefault` here would be a real bug -- and the tests must not leak an override into
    the rest of the process.
    """
    monkeypatch.setenv(NUM_CHUNKS_ENV, "999")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param([], DEFAULT_NUM_CHUNKS, id="default"),
        pytest.param(["--num-chunks", "8"], 8, id="explicit"),
        pytest.param(["--num-chunks=8"], 8, id="equals-form"),
        # The real parser's flags are all unknown to the mini-parser; parse_known_args must
        # swallow them rather than erroring or stopping at the first one it does not know.
        pytest.param(["--date", "20260806T00", "-vv"], DEFAULT_NUM_CHUNKS, id="unknown-only"),
        pytest.param(
            ["--date", "20260806T00", "--members", "4", "--num-chunks", "4", "--overwrite"],
            4,
            id="after-unknown",
        ),
        # `aifs single --help` still goes through here first; the mini-parser must not
        # intercept it and print its own (empty) help.
        pytest.param(["--help"], DEFAULT_NUM_CHUNKS, id="help"),
    ],
)
def test_num_chunks_reaches_the_environment_as_a_string(argv, expected):
    assert preresolve_num_chunks(argv) == expected
    assert os.environ[NUM_CHUNKS_ENV] == str(expected), (
        "anemoi reads the environment, not the return value"
    )


def test_ensemble_only_flags_are_rejected_by_the_deterministic_parser():
    """Accepting and ignoring `--members` would look like a working ensemble run."""
    single = build_parser(get_model("single"))
    for flag in ("--members", "--seed-offset"):
        with pytest.raises(SystemExit):
            single.parse_args([flag, "3"])

    args = build_parser(get_model("ens")).parse_args(["--members", "3", "--seed-offset", "10"])
    assert (args.members, args.seed_offset) == (3, 10)


def test_defaults_come_from_the_model_spec_rather_than_the_parser():
    """Hardcoding 360/10 here would make the registry a lie for any future variant."""
    single = dataclasses.replace(SINGLE, default_lead_time=42)
    assert build_parser(single).parse_args([]).lead_time == 42

    ens = dataclasses.replace(ENS, default_lead_time=48, default_members=7)
    args = build_parser(ens).parse_args([])
    assert (args.lead_time, args.members) == (48, 7)
