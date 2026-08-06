"""Streaming, non-blocking Zarr writer, shared by both AIFS variants.

The reference scripts hold a whole forecast in memory and write once. At 25 fields on the
0.25 deg grid a step costs ~104 MB, so a 50-day deterministic run needs ~21 GB before a
byte reaches disk, times the member count for an ensemble: fine on a 75 GB Slurm
allocation, not on a laptop. :class:`ForecastZarrWriter` instead consumes steps as the
model produces them, on a background thread, so peak memory tracks the *batch size* rather
than forecast length or ensemble size, and one batch's write overlaps the next inference.

The store is preallocated for the whole run, so every write is a plain region write -- no
append, no metadata rewriting, no read-back. Three dimensions fill incrementally:

* ``number`` -- one member at a time, sequentially (a single MPS device, unlike the
  reference's multi-GPU pool); the size-1 chunks *and* shards from
  :mod:`aifs_mps.zarr_layout` are what keep those writes independent. Absent entirely for
  deterministic models.
* ``prediction_timedelta`` -- 6-hourly outputs, buffered to one chunk (24 steps) so each
  region write is chunk-aligned.
* ``prediction_timedelta_daily`` -- daily aggregates, one slice per four steps from a
  running accumulator, so raw steps are never buffered at all.

Buffers therefore never scale with ensemble size. Steps cross the thread boundary as N320
arrays (~54 MB for 25 fields) rather than regridded lat/lon arrays (~104 MB), and the queue
is bounded, so a slow disk applies backpressure instead of growing the heap.

Measured against the batch path on a real 48 h forecast with BitRound disabled: 6-hourly
fields are *bit-identical*; daily aggregates differ by at most 1.5e-7 relative (~1 float32
ulp) because this accumulator sums in float64 before casting whereas the batch path coarsens
float32 -- below the store's lossy BitRound floor, and the more accurate of the two.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import dask.array as da
import numpy as np
import xarray as xr

from .config import DAILY_MAX
from .config import DAILY_MEAN
from .config import DAILY_MIN
from .config import DAILY_SUM
from .config import StoreSpec
from .regrid import LATITUDES
from .regrid import LATLON_SHAPE
from .regrid import LONGITUDES
from .regrid import to_latlon
from .zarr_layout import MODEL_STEP
from .zarr_layout import PT_DAILY_DESCRIPTION
from .zarr_layout import STEPS_PER_DAY
from .zarr_layout import complete_day_windows
from .zarr_layout import quiet_consolidated_metadata_warning
from .zarr_layout import v3_encoding

LOG = logging.getLogger(__name__)

__all__ = ["ForecastZarrWriter"]


class _Reducer(NamedTuple):
    """A daily reduction expressed incrementally, so no raw steps are ever buffered.

    ``identity`` seeds the accumulator, ``combine`` folds in one step, and ``finalize``
    turns the accumulator into the stored value once the day is complete.

    ``np.minimum``/``np.maximum`` propagate NaN, matching the additive reducers and the
    batch path's ``skipna=False``: a day with a missing sample has no valid extreme.
    """

    identity: float
    combine: Callable[[np.ndarray, np.ndarray], np.ndarray]
    finalize: Callable[[np.ndarray], np.ndarray] = lambda accumulated: accumulated


# Keyed by the config constants, not string literals, so an aggregation that config
# accepts but this table lacks raises in _reducer() instead of silently falling through.
_REDUCERS: dict[str, _Reducer] = {
    DAILY_MEAN: _Reducer(0.0, np.add, lambda accumulated: accumulated / STEPS_PER_DAY),
    DAILY_SUM: _Reducer(0.0, np.add),
    DAILY_MIN: _Reducer(np.inf, np.minimum),
    DAILY_MAX: _Reducer(-np.inf, np.maximum),
}


def _reducer(aggregation: str) -> _Reducer:
    try:
        return _REDUCERS[aggregation]
    except KeyError:
        raise ValueError(
            f"No streaming reducer for aggregation {aggregation!r}; "
            f"implemented: {', '.join(sorted(_REDUCERS))}"
        ) from None


class ForecastZarrWriter:
    """Consume forecast states and stream them into a Zarr store.

    Use as a context manager. Call :meth:`start_member` before each rollout, then
    :meth:`submit` per step; both return without waiting for the write. Exiting the block
    flushes, joins the worker and renames the staged store into place.

    ``n_members=None`` writes a deterministic store with no ``number`` dimension. The
    driver still calls ``start_member(0)`` once, so there is a single code path for both
    models.
    """

    def __init__(
        self,
        save_path: Path,
        init_date,
        n_steps: int,
        n_members: int | None,
        matrix,
        spec: StoreSpec,
        batch_steps: int | None = None,
        queue_size: int = 2,
    ) -> None:
        self.save_path = Path(save_path)
        self.partial_path = self.save_path.with_name(
            self.save_path.stem + "_partial" + self.save_path.suffix
        )
        self.init_date = init_date
        self.n_steps = n_steps
        self.matrix = matrix
        self.spec = spec
        self.encoding = spec.encoding
        # Read at call time so the module constant stays overridable (the tests run the
        # whole writer on a small stand-in grid).
        self.grid_shape = LATLON_SHAPE
        self.batch_steps = batch_steps or self.encoding.chunks["prediction_timedelta"]

        self.ensemble = n_members is not None
        self.n_members = 1 if n_members is None else n_members
        ens_dim = ("number",) if self.ensemble else ()
        self.native_dims = ("time", *ens_dim, "prediction_timedelta", "lat", "lon")
        self.daily_dims = ("time", *ens_dim, "prediction_timedelta_daily", "lat", "lon")

        self.native_outputs = spec.native
        self.daily_outputs = spec.daily
        self.sources = spec.source_fields

        self.windows = complete_day_windows(init_date, n_steps) if self.daily_outputs else []
        self._window_of_step = {
            i: day for day, (start, stop) in enumerate(self.windows) for i in range(start, stop)
        }

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="zarr-writer", daemon=True)

        self._current_member: int | None = None
        self._n_submitted = 0
        self._members_started: list[int] = []

        self._reset_member_state()

    def _reset_member_state(self) -> None:
        """Worker-owned buffers, cleared at every member boundary."""
        self._member: int | None = None
        self._native_buffer: list[np.ndarray] = []
        self._native_start = 0
        self._daily_buffer: list[np.ndarray] = []
        self._daily_start = 0
        self._accum: dict[str, np.ndarray] = {}
        self._accum_day: int | None = None
        self._accum_count = 0

    # -- lifecycle ---------------------------------------------------------------

    def __enter__(self) -> ForecastZarrWriter:
        self._prepare_store()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(failed=exc_type is not None)

    def _prepare_store(self) -> None:
        if self.partial_path.exists():
            LOG.warning("Partial output %s already exists. Removing it.", self.partial_path)
            shutil.rmtree(self.partial_path)
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)

        template = self._template()
        LOG.info(
            "Preallocating %s: %s%d steps x %d 6-hourly outputs, %d days x %d daily outputs",
            self.partial_path,
            f"{self.n_members} members x " if self.ensemble else "",
            self.n_steps,
            len(self.native_outputs),
            len(self.windows),
            len(self.daily_outputs),
        )
        with quiet_consolidated_metadata_warning():
            template.to_zarr(
                self.partial_path,
                zarr_format=3,
                mode="w",
                encoding=v3_encoding(template, self.encoding),
                compute=False,
            )

    def _template(self) -> xr.Dataset:
        """Lazy, all-zeros dataset with the final shapes -- writes metadata only."""
        init = np.datetime64(self.init_date).astype("datetime64[ns]")
        steps = ((np.arange(self.n_steps) + 1) * MODEL_STEP).astype("timedelta64[ns]")
        # Each day is labelled by its window end, i.e. the lead time of its last step.
        daily = np.array([steps[stop - 1] for _, stop in self.windows], dtype="timedelta64[ns]")

        coords = {
            "time": [init],
            "lat": LATITUDES,
            "lon": LONGITUDES,
            "prediction_timedelta": steps,
        }
        if self.ensemble:
            coords["number"] = np.arange(self.n_members, dtype="int64")
        if self.daily_outputs:
            coords["prediction_timedelta_daily"] = daily

        def _zeros(dims: tuple[str, ...], shape: tuple[int, ...]) -> da.Array:
            return da.zeros(
                shape,
                dtype="float32",
                chunks=tuple(
                    min(self.encoding.shards[d], s) for d, s in zip(dims, shape, strict=True)
                ),
            )

        ens_shape = (self.n_members,) if self.ensemble else ()
        native_shape = (1, *ens_shape, self.n_steps, *self.grid_shape)
        daily_shape = (1, *ens_shape, len(self.windows), *self.grid_shape)

        data_vars: dict[str, tuple] = {
            o.name: (self.native_dims, _zeros(self.native_dims, native_shape))
            for o in self.native_outputs
        }
        data_vars.update(
            {
                o.name: (self.daily_dims, _zeros(self.daily_dims, daily_shape))
                for o in self.daily_outputs
            }
        )

        ds = xr.Dataset(data_vars, coords=coords)

        # Region writes never touch attributes, so they must be set on the template.
        for output in self.spec.outputs:
            ds[output.name].attrs.update(output.attrs)
        if self.daily_outputs:
            ds["prediction_timedelta_daily"].attrs["description"] = PT_DAILY_DESCRIPTION
        return ds

    # -- producer side -----------------------------------------------------------

    def start_member(self, member: int) -> None:
        """Declare that subsequent :meth:`submit` calls belong to `member`."""
        self._raise_worker_error()
        if not 0 <= member < self.n_members:
            raise ValueError(
                f"Member {member} outside the preallocated range 0..{self.n_members - 1}"
            )
        if member in self._members_started:
            raise RuntimeError(f"Member {member} was already started")
        if self._current_member is not None and self._n_submitted != self.n_steps:
            raise RuntimeError(
                f"Member {self._current_member} submitted {self._n_submitted} of "
                f"{self.n_steps} steps before member {member} started"
            )
        self._members_started.append(member)
        self._current_member = member
        self._n_submitted = 0
        self._queue.put(("member", member))

    def submit(self, state: dict) -> None:
        """Queue one forecast state for the current member. Does not wait for the write."""
        self._raise_worker_error()
        if self._current_member is None:
            raise RuntimeError("start_member() must be called before submit()")
        if self._n_submitted >= self.n_steps:
            raise RuntimeError(
                f"Member {self._current_member} received more than {self.n_steps} steps"
            )
        payload = ("step", self._n_submitted, {f: np.asarray(state["fields"][f]) for f in self.sources})
        self._n_submitted += 1
        self._queue.put(payload)  # blocks only if the writer has fallen `queue_size` behind

    def close(self, failed: bool = False) -> None:
        """Flush, stop the worker, and move the staged store into place."""
        self._queue.put(None)
        self._thread.join()
        self._raise_worker_error()

        if failed:
            LOG.warning("Run failed; leaving staged store at %s", self.partial_path)
            return

        missing = sorted(set(range(self.n_members)) - set(self._members_started))
        if missing:
            raise RuntimeError(
                f"Store missing members after run: {missing}; "
                f"staged store left at {self.partial_path}"
            )
        if self._n_submitted != self.n_steps:
            raise RuntimeError(
                f"Member {self._current_member} submitted {self._n_submitted} of "
                f"{self.n_steps} steps; staged store left at {self.partial_path}"
            )
        if self.save_path.exists():
            shutil.rmtree(self.save_path)
        self.partial_path.rename(self.save_path)
        LOG.info("Forecast saved to %s", self.save_path)

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Zarr writer thread failed") from self._error

    # -- worker side -------------------------------------------------------------

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    self._flush_member()
                    return
                if item[0] == "member":
                    self._flush_member()
                    self._reset_member_state()
                    self._member = item[1]
                else:
                    self._consume(item[1], item[2])
        except BaseException as exc:  # re-raised on the producer thread
            self._error = exc
            LOG.exception("Zarr writer thread failed")
            # Drain so a blocked producer can make progress and observe the error.
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass

    def _flush_member(self) -> None:
        self._flush_native()
        self._flush_daily()

    def _consume(self, index: int, fields: dict[str, np.ndarray]) -> None:
        # Regrid each source once; several outputs may be derived from the same field.
        regridded = {
            source: to_latlon(fields[source], self.matrix, self.grid_shape)
            for source in self.sources
        }

        if self.native_outputs:
            self._native_buffer.append(
                np.stack([regridded[o.source] for o in self.native_outputs]).astype(np.float32)
            )
            if len(self._native_buffer) >= self.batch_steps:
                self._flush_native()

        day = self._window_of_step.get(index)
        if day is None:
            return  # step belongs to an incomplete edge day; dropped, as in the batch path

        if day != self._accum_day:
            self._accum = {
                o.name: np.full(
                    self.grid_shape, _reducer(o.aggregation).identity, dtype=np.float64
                )
                for o in self.daily_outputs
            }
            self._accum_day = day
            self._accum_count = 0

        for output in self.daily_outputs:
            self._accum[output.name] = _reducer(output.aggregation).combine(
                self._accum[output.name], regridded[output.source]
            )
        self._accum_count += 1

        if self._accum_count == STEPS_PER_DAY:
            self._daily_buffer.append(
                np.stack(
                    [
                        _reducer(o.aggregation).finalize(self._accum[o.name])
                        for o in self.daily_outputs
                    ]
                ).astype(np.float32)
            )
            self._accum = {}
            self._accum_day = None
            if len(self._daily_buffer) >= self.encoding.chunks["prediction_timedelta_daily"]:
                self._flush_daily()

    def _flush_native(self) -> None:
        if not self._native_buffer:
            return
        block = np.stack(self._native_buffer, axis=1)  # (output, step, lat, lon)
        start, stop = self._native_start, self._native_start + block.shape[1]
        names = [o.name for o in self.native_outputs]
        self._write_region(names, block, "prediction_timedelta", start, stop)
        self._native_start = stop
        self._native_buffer = []

    def _flush_daily(self) -> None:
        if not self._daily_buffer:
            return
        block = np.stack(self._daily_buffer, axis=1)  # (output, day, lat, lon)
        start, stop = self._daily_start, self._daily_start + block.shape[1]
        names = [o.name for o in self.daily_outputs]
        self._write_region(names, block, "prediction_timedelta_daily", start, stop)
        self._daily_start = stop
        self._daily_buffer = []

    def _write_region(
        self, names: list[str], block: np.ndarray, dim: str, start: int, stop: int
    ) -> None:
        if self._member is None:
            raise RuntimeError("No member is active; start_member() was never seen by the worker")
        ens_dim = ("number",) if self.ensemble else ()
        dims = ("time", *ens_dim, dim, "lat", "lon")
        expand = (np.newaxis, np.newaxis) if self.ensemble else (np.newaxis,)
        ds = xr.Dataset({name: (dims, block[i][expand]) for i, name in enumerate(names)})

        LOG.info(
            "Writing %s%s[%d:%d] (%d outputs)",
            f"member {self._member} " if self.ensemble else "",
            dim,
            start,
            stop,
            len(names),
        )
        region = {
            "time": slice(0, 1),
            dim: slice(start, stop),
            "lat": slice(None),
            "lon": slice(None),
        }
        if self.ensemble:
            region["number"] = slice(self._member, self._member + 1)

        with quiet_consolidated_metadata_warning():
            ds.to_zarr(self.partial_path, region=region)
