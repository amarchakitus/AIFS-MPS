"""Checkpoint download: the prompt, and the integrity of a 1-2.6 GB transfer.

Everything is mocked; these never touch the network.
"""

from __future__ import annotations

import pytest

from aifs_mps import weights
from aifs_mps.models import ENS
from aifs_mps.models import SINGLE

BODY = b"x" * 5000


class FakeResponse:
    def __init__(self, body=BODY, status=200, total=None, chunk=1000):
        self.content = body
        self.status_code = status
        self._chunk = chunk
        self.headers = {"Content-Length": str(total if total is not None else len(body))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        for i in range(0, len(self.content), self._chunk):
            yield self.content[i : i + self._chunk]


@pytest.fixture
def fake_http(monkeypatch):
    """Record requests and serve a small body, honouring Range like Hugging Face does."""
    calls = {"get": [], "head": []}

    def get(url, stream=False, headers=None, timeout=None):
        headers = headers or {}
        calls["get"].append((url, headers))
        rng = headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            return FakeResponse(BODY[start:], status=206)
        return FakeResponse()

    def head(url, allow_redirects=True, timeout=None):
        calls["head"].append(url)
        return FakeResponse(b"")

    monkeypatch.setattr(weights.requests, "get", get)
    monkeypatch.setattr(weights.requests, "head", head)
    return calls


@pytest.fixture
def answers(monkeypatch):
    """Drive the menu: pretend to be a terminal and feed canned keystrokes."""

    def setup(*keys):
        pending = list(keys)
        monkeypatch.setattr(weights.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _prompt="": pending.pop(0))

    return setup


# -- the menu --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected"),
    [("0", []), ("1", ["single", "ens"]), ("2", ["single"]), ("3", ["ens"])],
)
def test_each_menu_choice_selects_the_right_models(tmp_path, fake_http, answers, key, expected):
    answers(key)
    chosen = weights._ask([SINGLE, ENS], tmp_path)
    assert [s.name for s in chosen] == expected


def test_an_invalid_answer_reprompts_rather_than_downloading(tmp_path, fake_http, answers):
    """A typo must not be read as consent to pull 3.5 GB."""
    answers("yes", "9", "2")
    assert [s.name for s in weights._ask([SINGLE, ENS], tmp_path)] == ["single"]


def test_repeated_invalid_answers_give_up_without_downloading(tmp_path, fake_http, answers):
    answers("a", "b", "c")
    assert weights._ask([SINGLE, ENS], tmp_path) == []


def test_a_non_interactive_setup_never_blocks_on_the_prompt(tmp_path, fake_http, monkeypatch):
    """In CI or under `| tee`, input() would hang forever waiting on a terminal."""
    monkeypatch.setattr(weights.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input", lambda _p="": pytest.fail("prompted without a terminal")
    )
    assert weights._ask([SINGLE, ENS], tmp_path) == []


# -- deciding what to fetch ------------------------------------------------------


def test_present_checkpoints_are_never_re_downloaded(tmp_path, fake_http, monkeypatch):
    for spec in (SINGLE, ENS):
        (tmp_path / spec.checkpoint).write_bytes(b"already here")
    monkeypatch.setattr("builtins.input", lambda _p="": pytest.fail("prompted needlessly"))

    assert weights.ensure_checkpoints(directory=tmp_path, progress=False) == []
    assert not fake_http["get"], "no download should have been attempted"


def test_choosing_one_model_leaves_the_other_alone(tmp_path, fake_http, answers):
    answers("3")
    written = weights.ensure_checkpoints(directory=tmp_path, progress=False)
    assert [p.name for p in written] == [ENS.checkpoint]
    assert not (tmp_path / SINGLE.checkpoint).exists()


def test_a_partly_populated_directory_only_fetches_the_gap(tmp_path, fake_http, answers):
    (tmp_path / SINGLE.checkpoint).write_bytes(b"already here")
    answers("1")  # asked for both
    written = weights.ensure_checkpoints(directory=tmp_path, progress=False)
    assert [p.name for p in written] == [ENS.checkpoint], "the present one must be skipped"
    assert (tmp_path / SINGLE.checkpoint).read_bytes() == b"already here", "not overwritten"


def test_declining_downloads_nothing(tmp_path, fake_http, answers):
    answers("0")
    assert weights.ensure_checkpoints(directory=tmp_path, progress=False) == []
    assert not fake_http["get"]


def test_assume_no_skips_without_prompting(tmp_path, fake_http, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _p="": pytest.fail("prompted despite --no"))
    assert weights.ensure_checkpoints(directory=tmp_path, assume_yes=False, progress=False) == []
    assert not fake_http["get"]


def test_assume_yes_downloads_without_prompting(tmp_path, fake_http, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _p="": pytest.fail("prompted despite --yes"))
    written = weights.ensure_checkpoints(directory=tmp_path, assume_yes=True, progress=False)
    assert len(written) == 2


def test_the_weights_directory_is_created_if_absent(tmp_path, fake_http):
    target = tmp_path / "does" / "not" / "exist"
    weights.download_checkpoint(SINGLE, target, progress=False)
    assert (target / SINGLE.checkpoint).exists()


# -- transfer integrity ----------------------------------------------------------


def test_a_failed_download_leaves_nothing_at_the_final_path(tmp_path, monkeypatch):
    """The critical property: a truncated checkpoint must never look complete.

    It would load as a file and then fail somewhere inside unpickling, with an error that
    points at the model rather than at the download.
    """

    def truncated(url, stream=False, headers=None, timeout=None):
        return FakeResponse(BODY[:1500], total=len(BODY))  # server claims more than it sends

    monkeypatch.setattr(weights.requests, "get", truncated)

    with pytest.raises(RuntimeError, match=r"expected .* bytes but got"):
        weights.download_checkpoint(SINGLE, tmp_path, progress=False)

    assert not (tmp_path / SINGLE.checkpoint).exists()
    assert not list(tmp_path.glob("*.partial")), "the partial file must be cleaned up too"


def test_an_interrupted_download_resumes_instead_of_restarting(tmp_path, fake_http):
    """3.5 GB over a flaky link: restarting from zero every time never finishes."""
    partial = tmp_path / (SINGLE.checkpoint + ".partial")
    partial.write_bytes(BODY[:2000])

    weights.download_checkpoint(SINGLE, tmp_path, progress=False)

    _url, headers = fake_http["get"][0]
    assert headers.get("Range") == "bytes=2000-", "must ask for the remainder only"
    assert (tmp_path / SINGLE.checkpoint).read_bytes() == BODY, "resumed file must be correct"


def test_a_server_ignoring_range_restarts_cleanly(tmp_path, monkeypatch):
    """Answering 200 to a Range request means the body is the WHOLE file; appending it to
    the partial prefix would silently produce a corrupt checkpoint."""
    monkeypatch.setattr(
        weights.requests,
        "get",
        lambda url, stream=False, headers=None, timeout=None: FakeResponse(BODY, status=200),
    )
    (tmp_path / (SINGLE.checkpoint + ".partial")).write_bytes(BODY[:2000])

    weights.download_checkpoint(SINGLE, tmp_path, progress=False)
    assert (tmp_path / SINGLE.checkpoint).read_bytes() == BODY


def test_each_model_resolves_to_its_own_checkpoint_url():
    """Guards a copy-paste slip between the two registry entries."""
    urls = {spec.name: weights.checkpoint_url(spec) for spec in (SINGLE, ENS)}
    for spec in (SINGLE, ENS):
        assert urls[spec.name].startswith("https://huggingface.co/ecmwf/")
        assert spec.checkpoint in urls[spec.name]
    assert urls["single"] != urls["ens"]


def test_the_endpoint_can_be_pointed_at_a_mirror(monkeypatch):
    """HF_ENDPOINT is the Hugging Face convention; honouring it costs nothing and is the
    difference between usable and unusable behind a proxy."""
    monkeypatch.setattr(weights, "HF_ENDPOINT", "https://mirror.example")
    assert weights.checkpoint_url(SINGLE).startswith("https://mirror.example/ecmwf/")
