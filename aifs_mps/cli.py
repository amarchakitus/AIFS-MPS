#!/usr/bin/env python
"""One forecast driver for both AIFS variants.

The deterministic and ensemble runs differ only in whether there is more than one member
and whether a seed is set before each rollout, so they share this module entirely; what
each model *is* lives in :mod:`aifs_mps.models`.

Import order in :func:`main` is load-bearing; see the comment there.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path

from .models import ModelSpec
from .models import get_model

LOG = logging.getLogger("aifs")

STEP_HOURS = 6  # both AIFS v2 models step 6-hourly

# Chunking is the single most effective memory knob. Measured on an M4 Max, one 6-hourly
# AIFS-ENS step: 1 chunk -> 88 GB peak driver allocation, 16 -> 29 GB, 32 -> 20 GB,
# 64 -> 17 GB, with no slowdown (chunked is actually *faster* than unchunked).
DEFAULT_NUM_CHUNKS = 32


def _valid_date(text: str) -> datetime.datetime:
    for fmt in ("%Y%m%dT%H", "%Y-%m-%dT%H", "%Y-%m-%d %H", "%Y%m%d%H", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Cannot parse date {text!r}; try 20260806T00")


def preresolve_num_chunks(argv: list[str]) -> int:
    """Set ANEMOI_INFERENCE_NUM_CHUNKS before anemoi.models is imported.

    anemoi reads it into module-level constants at *import* time
    (anemoi/models/layers/{mapper,block}.py), so by the time argparse runs it is too late.
    Hence this mini-parse ahead of the real parser.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS)
    num_chunks = pre.parse_known_args(argv)[0].num_chunks
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(num_chunks)
    return num_chunks


def build_parser(spec: ModelSpec) -> argparse.ArgumentParser:
    # Imported lazily: these pull in torch/anemoi, which must happen after the patches.
    from .config import DEFAULT_CONFIG
    from .opendata import DEFAULT_SOURCES
    from .opendata import SOURCES
    from .patches import ATTENTION_DTYPES
    from .paths import forecast_dir
    from .paths import input_state_dir
    from .paths import lsm_path
    from .paths import regrid_dir
    from .paths import weights_dir

    p = argparse.ArgumentParser(
        prog=f"aifs-{spec.name}",
        description=f"Run {spec.description} on Apple silicon (MPS/Metal).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ic = p.add_argument_group("initial conditions")
    ic.add_argument("--date", type=_valid_date, help="init time, e.g. 20260806T00 (default: latest)")
    ic.add_argument(
        "--input-state", type=Path, help="use this pickled state instead of the shared cache"
    )
    ic.add_argument(
        "--source",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        choices=SOURCES,
        metavar="MIRROR",
        help="open-data mirrors to try, in order; each request falls through on failure",
    )
    ic.add_argument(
        "--no-cache-input-state",
        action="store_true",
        help=f"do not write the downloaded state to {input_state_dir()}",
    )
    ic.add_argument("--lsm", type=Path, default=lsm_path(), help="N320 land-sea mask GRIB")
    ic.add_argument("--regrid-dir", type=Path, default=regrid_dir(), help="interpolation matrices")

    fc = p.add_argument_group("forecast")
    fc.add_argument("--lead-time", type=int, default=spec.default_lead_time, help="hours")
    if spec.ensemble:
        fc.add_argument("--members", type=int, default=spec.default_members, help="ensemble size")
        fc.add_argument(
            "--seed-offset",
            type=int,
            default=0,
            help="added to the member index to form the torch seed, so a second run can "
            "extend the ensemble with different members",
        )
    fc.add_argument(
        "--checkpoint", type=Path, default=weights_dir() / spec.checkpoint, help="model weights"
    )
    fc.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    fc.add_argument("--precision", help="override autocast precision, e.g. 32 (default: from ckpt)")
    fc.add_argument(
        "--attention-dtype",
        choices=list(ATTENTION_DTYPES),
        default="float32",
        help="dtype for attention scores/softmax, independent of --precision",
    )
    fc.add_argument("--attention-block", type=int, help="banded-attention query block size")
    fc.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help="mapper/processor chunking; the main memory knob (1 peaks at ~88 GB/step)",
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "--output",
        type=Path,
        help=f"zarr store (default: {forecast_dir(spec.name)}/init_<date>.zarr)",
    )
    out.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML store spec: which variables, their units and time aggregations",
    )
    out.add_argument(
        "--save-fields",
        nargs="+",
        metavar="FIELD",
        help="restrict to these config variables, or 'all' for every model field 6-hourly",
    )
    out.add_argument(
        "--no-daily-aggregates",
        action="store_true",
        help="store every configured variable 6-hourly instead of aggregating",
    )
    out.add_argument("--write-batch-steps", type=int, help="steps buffered per region write")
    out.add_argument("--overwrite", action="store_true", help="rerun even if the store exists")
    out.add_argument("-v", "--verbose", action="count", default=0)
    return p


def _free_mps(device: str) -> None:
    """Return cached-but-unused MPS blocks to the system.

    The MPS caching allocator, not live tensors, is what runs the machine out of memory:
    live usage sits around 3 GB while reserved can balloon past 80 GB without this.
    """
    import torch

    if device == "mps":
        torch.mps.empty_cache()


def _mps_mem(device: str) -> str:
    import torch

    if device != "mps":
        return ""
    gb = 1 / 2**30
    return (
        f" [mps live={torch.mps.current_allocated_memory() * gb:.1f} GB"
        f" reserved={torch.mps.driver_allocated_memory() * gb:.1f} GB]"
    )


def run(model_spec: ModelSpec, args: argparse.Namespace) -> Path:
    """Run `model_spec` end to end and return the path to the finished store."""
    import torch

    from . import patches
    from .config import load_store_spec
    from .config import native_only_spec
    from .models import check_runtime
    from .opendata import latest_date
    from .opendata import load_or_build_input_state
    from .opendata import select_for_model
    from .patches import resolve_attention_dtype
    from .paths import forecast_dir
    from .regrid import n320_to_latlon_matrix
    from .zarr_stream import ForecastZarrWriter

    check_runtime(model_spec)

    LOG.info("torch %s | mps available=%s", torch.__version__, torch.backends.mps.is_available())
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine")

    patches.attention.DEFAULT_ATTN_DTYPE = resolve_attention_dtype(args.attention_dtype)
    if args.attention_block:
        patches.attention.DEFAULT_BLOCK = args.attention_block

    members = getattr(args, "members", 1) if model_spec.ensemble else 1
    if members < 1:
        raise SystemExit("--members must be at least 1")
    if args.lead_time % STEP_HOURS:
        raise SystemExit(f"--lead-time must be a multiple of {STEP_HOURS} h")
    n_steps = args.lead_time // STEP_HOURS

    # Resolve the date and check the output *before* retrieval: retrieval is the expensive
    # part, and there is no point paying it only to bail out.
    if args.input_state:
        import pickle

        with Path(args.input_state).open("rb") as f:
            superset = pickle.load(f)
        date = superset["date"]
    else:
        date = args.date or latest_date(args.source)
        superset = None

    save_path = args.output or forecast_dir(model_spec.name) / f"init_{date.strftime('%Y%m%dT%H')}.zarr"
    if save_path.exists() and not args.overwrite:
        raise SystemExit(f"{save_path} already exists; pass --overwrite to rerun")

    if superset is None:
        import earthkit.data as ekd

        ekd.config.set({"cache-policy": "user"})
        t_ic = time.time()
        superset = load_or_build_input_state(
            date,
            sources=args.source,
            cache=not args.no_cache_input_state,
            lsm=args.lsm,
            regrid_directory=args.regrid_dir,
        )
        LOG.info("Initial conditions ready in %.0fs", time.time() - t_ic)

    input_state = select_for_model(superset, model_spec)
    del superset

    from anemoi.inference.runners.simple import SimpleRunner

    runner = SimpleRunner(
        str(args.checkpoint), device=args.device, precision=args.precision, verbosity=args.verbose
    )
    LOG.info(
        "%s | autocast=%s attention=%s block=%d num_chunks=%s",
        model_spec.pretty,
        runner.autocast,
        args.attention_dtype,
        patches.attention.DEFAULT_BLOCK,
        os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS"),
    )

    matrix = n320_to_latlon_matrix(args.regrid_dir)

    spec = load_store_spec(args.config)
    LOG.info("Store spec: %s", args.config)
    if args.save_fields == ["all"]:
        spec = None  # every model field, resolved from the first state below
    elif args.save_fields:
        spec = spec.subset(args.save_fields)
    if spec is not None and args.no_daily_aggregates:
        spec = spec.without_daily()

    t0 = time.time()
    # Start member 0 before the writer exists so `--save-fields all` can be resolved from
    # a real state rather than guessed.
    torch.manual_seed(getattr(args, "seed_offset", 0))
    steps = runner.run(input_state=input_state, lead_time=args.lead_time)
    first = next(steps)
    if spec is None:
        spec = native_only_spec(sorted(first["fields"]))

    by_aggregation: dict[str, int] = {}
    for output in spec.outputs:
        by_aggregation[output.aggregation] = by_aggregation.get(output.aggregation, 0) + 1
    LOG.info(
        "%s: %s%d steps, %d outputs from %d fields (%s)",
        model_spec.pretty,
        f"{members} members x " if model_spec.ensemble else "",
        n_steps,
        len(spec.outputs),
        len(spec.source_fields),
        ", ".join(f"{n} {a}" for a, n in sorted(by_aggregation.items())),
    )

    with ForecastZarrWriter(
        save_path,
        date,
        n_steps,
        members if model_spec.ensemble else None,
        matrix,
        spec,
        batch_steps=args.write_batch_steps,
    ) as writer:
        for member in range(members):
            member_t0 = time.time()
            writer.start_member(member)

            if member == 0:
                writer.submit(first)
                del first
                member_steps = steps
            else:
                # AIFS-ENS v2 is stochastic; the seed set immediately before the rollout is
                # what makes this member differ from the last.
                torch.manual_seed(args.seed_offset + member)
                member_steps = runner.run(input_state=input_state, lead_time=args.lead_time)

            for state in member_steps:
                writer.submit(state)
                _free_mps(args.device)

            _free_mps(args.device)
            if model_spec.ensemble:
                LOG.info(
                    "Member %d/%d done in %.1fs (elapsed %.1fs)%s",
                    member + 1,
                    members,
                    time.time() - member_t0,
                    time.time() - t0,
                    _mps_mem(args.device),
                )

    LOG.info("Complete: %d steps in %.1fs -> %s", n_steps * members, time.time() - t0, save_path)
    return save_path


def main(model: str, argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    spec = get_model(model)

    # 1. num_chunks into the environment, 2. MPS patches, 3. everything else. Both steps
    #    must precede any import of anemoi.models or any torch.load of a checkpoint.
    preresolve_num_chunks(argv)
    from . import patches

    patches.apply_all()

    args = build_parser(spec).parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    run(spec, args)


def main_single() -> None:
    main("single")


def main_ens() -> None:
    main("ens")
