"""Mirror fallback and the shared initial-condition cache.

Both behaviours here exist because retrieval is slow and flaky. Open data is served by
four mirrors that fail in different ways -- throttled, or simply not having published the
newest run yet -- so a request must walk the list until one answers, and the error a user
finally sees has to say *which* mirror failed *how*, because "throttled" and "not
published yet" call for different fixes. The cache is the other half: one superset state
is ~500 MB and takes minutes to fetch, and the promise of the design is that a given init
date is downloaded once and then reused by both models.

Nothing here touches the network: `ekd.from_source` and `OpendataClient` are replaced,
and the cache is redirected into tmp_path via AIFS_MPS_IC_DIR.
"""

from __future__ import annotations

import datetime
import pickle

import pytest

from aifs_mps import opendata
from aifs_mps.opendata import _from_source
from aifs_mps.opendata import latest_date
from aifs_mps.opendata import load_or_build_input_state
from aifs_mps.paths import input_state_path

DATE = datetime.datetime(2026, 8, 6, 6)


class Recorder:
    """Stands in for a mirror-dispatching callable, remembering who was asked and when."""

    def __init__(self, failures: dict[str, str], result: object = "fields"):
        self.failures = failures
        self.result = result
        self.tried: list[str] = []
        self.last_kwargs: dict = {}

    def __call__(self, source: str):
        self.tried.append(source)
        if source in self.failures:
            raise RuntimeError(self.failures[source])
        return self.result


@pytest.fixture
def fake_ekd(monkeypatch):
    """Replace `ekd.from_source` with a recorder keyed on the `source=` mirror."""

    def install(recorder):
        def from_source(name, *, source, **kwargs):
            assert name == "ecmwf-open-data"
            recorder.last_kwargs = kwargs
            return recorder(source)

        monkeypatch.setattr(opendata.ekd, "from_source", from_source)
        return recorder

    return install


@pytest.fixture
def fake_client(monkeypatch):
    """Replace `OpendataClient` so `.latest()` succeeds or fails per mirror."""

    def install(recorder):
        class Client:
            def __init__(self, source):
                self.source = source

            def latest(self):
                return recorder(self.source)

        monkeypatch.setattr(opendata, "OpendataClient", Client)
        return recorder

    return install


# -- mirror fallback: _from_source -----------------------------------------------


def test_mirrors_are_tried_in_order_until_one_answers(fake_ekd):
    """Order is a measured preference (see DEFAULT_SOURCES), not decoration: shuffling it
    or short-circuiting on the wrong mirror costs minutes per request."""
    recorder = fake_ekd(Recorder({"azure": "503 Slow Down", "ecmwf": "connection reset"}))

    result = _from_source(("azure", "ecmwf", "aws"), param=["2t"])

    assert result == "fields"
    assert recorder.tried == ["azure", "ecmwf", "aws"]
    assert recorder.last_kwargs == {"param": ["2t"]}, "request arguments must survive fallback"


def test_a_succeeding_mirror_is_not_followed_by_the_others(fake_ekd):
    """Retrieval is one request per parameter batch; retrying every mirror on success would
    multiply an already slow download by the number of mirrors."""
    recorder = fake_ekd(Recorder({}))

    _from_source(("azure", "ecmwf", "aws"))

    assert recorder.tried == ["azure"]


def test_total_failure_names_every_mirror_and_its_reason(fake_ekd):
    """A user has to tell "throttled" (retry later) from "not published yet" (pass --date);
    collapsing the failures into one generic message loses exactly that distinction."""
    fake_ekd(
        Recorder(
            {
                "azure": "503 Slow Down",
                "ecmwf": "404 not published yet",
                "aws": "connection reset",
            }
        )
    )

    with pytest.raises(RuntimeError) as excinfo:
        _from_source(("azure", "ecmwf", "aws"))

    message = str(excinfo.value)
    for mirror, reason in [
        ("azure", "503 Slow Down"),
        ("ecmwf", "404 not published yet"),
        ("aws", "connection reset"),
    ]:
        assert mirror in message
        assert reason in message


# -- mirror fallback: latest_date ------------------------------------------------


def test_latest_date_falls_through_to_the_next_mirror(fake_client):
    """Mirrors publish a new run at different times, so the first one asked is routinely
    the one that has not got it yet."""
    recorder = fake_client(Recorder({"azure": "not published yet"}, result=DATE))

    assert latest_date(("azure", "ecmwf", "aws")) == DATE
    assert recorder.tried == ["azure", "ecmwf"], "the answering mirror must end the search"


def test_latest_date_reports_why_each_mirror_failed(fake_client):
    fake_client(Recorder({"azure": "503 Slow Down", "ecmwf": "timed out"}, result=DATE))

    with pytest.raises(RuntimeError) as excinfo:
        latest_date(("azure", "ecmwf"))

    message = str(excinfo.value)
    assert "azure: 503 Slow Down" in message
    assert "ecmwf: timed out" in message


def test_a_bare_source_string_is_one_mirror_not_five(fake_client):
    """`--source azure` arrives as a string; iterating it would ask for mirrors 'a', 'z'..."""
    recorder = fake_client(Recorder({}, result=DATE))

    assert latest_date("azure") == DATE
    assert recorder.tried == ["azure"]


# -- the shared initial-condition cache ------------------------------------------


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AIFS_MPS_IC_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def builder(monkeypatch):
    """Install a stand-in for `build_input_state` and count its invocations."""

    def install(state=None, error=None):
        calls = []

        def build_input_state(**kwargs):
            calls.append(kwargs)
            if error is not None:
                raise error
            return state

        monkeypatch.setattr(opendata, "build_input_state", build_input_state)
        return calls

    return install


STATE = {"date": DATE, "fields": {"2t": [1.0, 2.0]}}


def test_a_cached_date_is_never_downloaded_again(cache_dir, builder):
    """This is the whole point of the shared cache: run single, then ens, and the second
    run must not touch the network."""
    calls = builder(error=AssertionError("a cache hit must not rebuild the state"))
    with input_state_path(DATE).open("wb") as f:
        pickle.dump(STATE, f)

    assert load_or_build_input_state(DATE) == STATE
    assert calls == []


def test_a_miss_builds_once_and_leaves_a_reusable_cache(cache_dir, builder):
    calls = builder(state=STATE)

    assert load_or_build_input_state(DATE) == STATE
    assert len(calls) == 1

    builder(error=AssertionError("the second run should have hit the cache"))
    assert load_or_build_input_state(DATE) == STATE


def test_caching_can_be_turned_off_without_disabling_the_build(cache_dir, builder):
    """`--no-cache` is for one-off dates; it must not litter the shared cache directory."""
    calls = builder(state=STATE)

    assert load_or_build_input_state(DATE, cache=False) == STATE
    assert len(calls) == 1
    assert list(cache_dir.iterdir()) == []


def test_an_interrupted_write_is_not_mistaken_for_a_valid_cache(cache_dir, builder, monkeypatch):
    """A ~500 MB pickle takes a while to write; if the process dies partway the truncated
    file must not appear at the final path, where the next run would load it and fail deep
    inside unpickling instead of simply downloading again."""
    builder(state=STATE)
    real_dump = opendata.pickle.dump
    interrupted = []

    def die_on_first_write(obj, file, *args, **kwargs):
        if not interrupted:
            interrupted.append(True)
            raise OSError("no space left on device")
        return real_dump(obj, file, *args, **kwargs)

    monkeypatch.setattr(opendata.pickle, "dump", die_on_first_write)

    with pytest.raises(OSError, match="no space left"):
        load_or_build_input_state(DATE)

    assert not input_state_path(DATE).exists()

    # ...and the next run must genuinely rebuild rather than read the debris.
    calls = builder(state=STATE)
    assert load_or_build_input_state(DATE) == STATE
    assert len(calls) == 1
