"""Fetch the model checkpoints from Hugging Face.

Downloads stream to a ``.partial`` and are renamed only once the size matches
``Content-Length``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import requests

from .models import MODELS
from .models import ModelSpec
from .paths import weights_dir

LOG = logging.getLogger(__name__)

__all__ = [
    "checkpoint_path",
    "checkpoint_url",
    "download_checkpoint",
    "ensure_checkpoints",
    "missing_checkpoints",
]

CHUNK_BYTES = 8 * 1024 * 1024
TIMEOUT = 60

# Checkpoint file source.
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HF_REVISION = os.environ.get("AIFS_MPS_HF_REVISION", "main")


def checkpoint_url(spec: ModelSpec) -> str:
    """Resolve URL for `spec`'s checkpoint."""
    return f"{HF_ENDPOINT}/{spec.hf_repo}/resolve/{HF_REVISION}/{spec.checkpoint}?download=true"


def checkpoint_path(spec: ModelSpec, directory: Path | None = None) -> Path:
    return (directory or weights_dir()) / spec.checkpoint


def missing_checkpoints(
    specs=None, directory: Path | None = None
) -> list[ModelSpec]:
    specs = list(MODELS.values()) if specs is None else list(specs)
    return [s for s in specs if not checkpoint_path(s, directory).exists()]


def _human(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def download_checkpoint(
    spec: ModelSpec, directory: Path | None = None, progress: bool = True
) -> Path:
    """Download one checkpoint, resuming a partial file if there is one."""
    directory = directory or weights_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = checkpoint_path(spec, directory)
    partial = target.with_suffix(target.suffix + ".partial")

    resume_from = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    response = requests.get(checkpoint_url(spec), stream=True, headers=headers, timeout=TIMEOUT)
    if resume_from and response.status_code == 200:
        # Server ignored the Range request; start over rather than append to a prefix.
        LOG.info("%s: server does not support resume, restarting", spec.checkpoint)
        resume_from = 0
        partial.unlink()
    elif resume_from:
        LOG.info("%s: resuming from %s", spec.checkpoint, _human(resume_from))
    response.raise_for_status()

    total = int(response.headers.get("Content-Length", 0)) + resume_from
    LOG.info("Downloading %s (%s) -> %s", spec.checkpoint, _human(total), directory)

    written = resume_from
    with partial.open("ab" if resume_from else "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            f.write(chunk)
            written += len(chunk)
            if progress and total:
                pct = 100 * written / total
                print(
                    f"\r  {spec.checkpoint}  {pct:5.1f}%  {_human(written)} / {_human(total)}",
                    end="",
                    flush=True,
                )
    if progress:
        print()

    if total and written != total:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{spec.checkpoint}: expected {total} bytes but got {written}. "
            "The partial file has been removed; re-run to try again."
        )

    partial.rename(target)
    LOG.info("Saved %s", target)
    return target


def _remote_size(spec: ModelSpec) -> int:
    try:
        head = requests.head(checkpoint_url(spec), allow_redirects=True, timeout=TIMEOUT)
        return int(head.headers.get("Content-Length", 0))
    except requests.RequestException:
        return 0


def _ask(specs: list[ModelSpec], directory: Path | None) -> list[ModelSpec]:
    """Offer the numbered menu and return the models the user chose."""
    by_name = {s.name: s for s in specs}
    single, ens = by_name.get("single"), by_name.get("ens")

    print("\nModel checkpoints:")
    for spec in specs:
        path = checkpoint_path(spec, directory)
        state = "present" if path.exists() else f"missing  ({_human(_remote_size(spec))})"
        print(f"  {spec.checkpoint:28s} {state}")

    if not sys.stdin.isatty():
        # Non-interactive (CI, piped input): never hang on a prompt, and never start a
        # multi-GB download nobody asked for.
        print("\n  stdin is not a terminal; skipping download.")
        print("  Use --yes to download, --model to pick one, --no to silence.")
        return []

    choices = {"0": [], "1": specs}
    menu = ["  0) none", "  1) both"]
    if single is not None:
        choices["2"] = [single]
        menu.append("  2) AIFS Single v2 only")
    if ens is not None:
        choices["3"] = [ens]
        menu.append("  3) AIFS-ENS v2 only")

    print("\nDownload from Hugging Face?")
    print("\n".join(menu))
    valid = "/".join(sorted(choices))
    for _ in range(3):
        try:
            answer = input(f"  [{valid}]: ").strip()
        except EOFError:
            return []
        if answer in choices:
            return choices[answer]
        print(f"  Please enter one of {valid}.")
    print("  No valid choice; skipping.")
    return []


def ensure_checkpoints(
    specs=None,
    directory: Path | None = None,
    assume_yes: bool | None = None,
    force: bool = False,
    progress: bool = True,
) -> list[Path]:
    """Make sure the checkpoints are present, asking first unless told otherwise.

    `assume_yes` of None prompts; True downloads without asking; False skips.
    Returns the paths downloaded (empty if everything was already there or declined).
    """
    specs = list(MODELS.values()) if specs is None else list(specs)

    if assume_yes is False:
        missing = missing_checkpoints(specs, directory)
        LOG.info("Skipping checkpoint download; %d missing", len(missing))
        return []

    if force:
        chosen = specs if assume_yes else _ask(specs, directory)
    elif not missing_checkpoints(specs, directory):
        LOG.info("Checkpoints already present: %s", ", ".join(s.checkpoint for s in specs))
        return []
    else:
        chosen = specs if assume_yes else _ask(specs, directory)

    # Honour the choice but never re-download something already on disk.
    todo = chosen if force else [s for s in chosen if not checkpoint_path(s, directory).exists()]
    if not todo:
        if chosen:
            LOG.info("Nothing to do; the selected checkpoints are already present.")
        else:
            print("  Skipped. Fetch later with:  ./aifs weights")
        return []

    return [download_checkpoint(spec, directory, progress) for spec in todo]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        prog="aifs-weights", description="Download AIFS checkpoints from Hugging Face."
    )
    p.add_argument("--model", choices=[*MODELS, "all"], default="all")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--yes", action="store_true", help="download without asking")
    group.add_argument("--no", action="store_true", help="never download; just report")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--weights-dir", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    specs = list(MODELS.values()) if args.model == "all" else [MODELS[args.model]]
    assume_yes = True if args.yes else (False if args.no else None)
    ensure_checkpoints(specs, args.weights_dir, assume_yes=assume_yes, force=args.force)
