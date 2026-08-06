# AIFS-MPS

Run ECMWF's **AIFS Single v2** (deterministic) and **AIFS-ENS v2** (ensemble) locally on an
Apple-silicon Mac, on the Metal GPU. Both models share an initial-condition cache and a consistent
Zarr output layout.

```bash
./aifs setup                        # create both runtime environments
./aifs single --lead-time 48        # deterministic
./aifs ens --members 10             # 10-member ensemble
./aifs single --help
```

The checkpoints are not in the repo. Download them from <https://huggingface.co/ecmwf> and
put them in `weights/`:

```
weights/aifs-single-mse-2.0.ckpt    # AIFS Single v2
weights/aifs-ens-crps-2.0.ckpt      # AIFS-ENS v2
```

Either name can be overridden with `--checkpoint`, or the directory by setting
`AIFS_MPS_WEIGHTS_DIR`.

If no `--date` is specified, the latest available IFS data from ECMWF Open Data are used as initial conditions. Outputs:

```
input_states/input_state_20260806T06.pkl   # shared by both models
forecasts/single/init_20260806T06.zarr     # (time, prediction_timedelta, lat, lon)
forecasts/ens/init_20260806T06.zarr        # (time, number, prediction_timedelta, lat, lon)
```

## Measured performance

16" MacBook Pro; M4 Max, 128 GB, macOS 26.6, torch 2.7.0, `--num-chunks 32`:

| | AIFS Single v2 | AIFS-ENS v2 |
|---|---|---|
| per 6 h step | ~6.5 s | ~12.5 s |
| 48 h forecast | 57 s | ~100 s per member |
| peak MPS memory | ~9 GB reserved | 8.8 GB reserved / 3.3 GB live |
| autocast (from checkpoint) | fp16 | bf16 |

Retrieving one initial state takes ~180 s from azure. It is then reused by both models.

## Why two environments

The two checkpoints require different `anemoi-models` versions:

- The Single v2 checkpoint pickles references to `anemoi.models.layers.chunk`, a module
  **deleted in 0.11.x**. Loading it under 0.11.2 fails with `ModuleNotFoundError`.
- The ENS v2 checkpoint needs `anemoi.models.layers.{ensemble,sparse_projector}` and
  `AnemoiEnsModelEncProcDec`, **none of which exist in 0.9.3**.

Neither version can load both checkpoints, so they cannot share a virtualenv. The repo
therefore has one shared library and two thin runtime projects that pin only the version
they need:

```
aifs_mps/                      all shared code — one copy
runtimes/single/pyproject.toml anemoi-models==0.9.3  + aifs-mps (editable path dep)
runtimes/ens/pyproject.toml    anemoi-models==0.11.2 + aifs-mps (editable path dep)
aifs                           dispatcher: picks the right environment
```

`./aifs single …` and `./aifs ens …` hide the split entirely. Both environments coexist, so
switching models costs nothing and you can run them concurrently. If you prefer no wrapper:

```bash
uv run --project runtimes/ens aifs-ens --members 4
```

Running a model in the wrong environment is caught up front by `models.check_runtime` with
a message telling you what to run, rather than failing inside the unpickler.

## One initial state cached for both models

The models require *different* input fields, so retrieving per-model would download the same
date twice. Instead `aifs_mps/opendata.py` retrieves the superset once, caches one file
per init date in `input_states/`, and each model drops what it does not use:

| | fields | drops |
|---|---|---|
| superset (cached) | 113 | — |
| AIFS Single v2 | 97 | `q_10`, `q_50`, and all 14 `w_*` levels |
| AIFS-ENS v2 | 112 | `q_10` |

Cache location is overridable by setting `AIFS_MPS_IC_DIR`; so are `AIFS_MPS_WEIGHTS_DIR`,
`AIFS_MPS_SUPPORT_DIR`, `AIFS_MPS_REGRID_DIR`, `AIFS_MPS_LSM` and `AIFS_MPS_FORECAST_DIR`.

## Memory

The MPS **caching allocator**, not live tensors, causes RAM exhaustion: live
usage sits ~3 GB while reserved can pass 80 GB. Specifying `--num-chunks` fixes this issue. Measured for one AIFS-ENS step:

| `--num-chunks` | peak driver memory | step time |
|---|---|---|
| 1 (anemoi default) | **88 GB** | 17–26 s |
| 16 | 29 GB | 13 s |
| **32 (default)** | **20 GB** | 12 s |
| 64 | 17 GB | 13 s |

`torch.mps.empty_cache()` runs after every step, holding the steady state near 8.8 GB.

`ANEMOI_INFERENCE_NUM_CHUNKS` is read into module-level constants when `anemoi.models` is
*imported*, so `cli.preresolve_num_chunks` resolves `--num-chunks` before any anemoi import.
That ordering is load-bearing.

## The MPS workarounds

`aifs_mps/patches/`, applied by `apply_all()` before the checkpoint loads. Each patch
detects whether it applies to the anemoi version in the current interpreter, so there is
one call site for both runtimes.

| patch | Single | ENS | why |
|---|---|---|---|
| `stubs` | ✓ | ✓ | `flash_attn` (both) and Triton (ENS) can't be imported on macOS, but the pickles name them |
| `attention` | ✓ | ✓ | flash-attn → banded SDPA that fits in memory |
| `graph_transformer` | – | ✓ | ENS pickles a Triton kernel; reroute to anemoi's own PyG backend |
| `sparse_projector` | – | ✓ | ENS's noise projection is a sparse matmul, which MPS has no kernel for |

**Banded attention.** Anemoi's own SDPA fallback builds a dense `seq_len × seq_len` mask;
on the o96 hidden mesh (40 320 tokens) that is 52 GB of fp16 scores. Both models use
sliding-window attention (`window_size=1120`), which is exactly the band `|i−j| ≤ 1120`, so
it is evaluated block by block — O(seq·window), numerically equivalent to flash-attn.

**The 2³² trap.** `scaled_dot_product_attention` on MPS returns *silently wrong numbers*
once `heads × q_len × k_len` exceeds 2³² (a 32-bit indexing overflow torch does not check).
End to end that showed up as a 6.3 K error in forecast 2 m temperature. `_safe_block()`
clamps for it.

**Triton.** anemoi-models 0.11.2 ships a Triton GraphTransformer kernel and the ENS
checkpoint was trained with it selected; the real module *raises on import* with no macOS
build, so the checkpoint could not even be opened. Anemoi's own PyG fallback lives in
`__init__`, which never runs when a whole model is unpickled — so the module is stubbed and
`apply_gt` is routed onto that same branch. `GraphTransformerConv` is parameter-free, so
nothing is lost.

**Sparse noise projection.** ENS projects noise onto the hidden mesh through a sparse COO
matrix (40320 × 5248, ~2.0M nnz); MPS has no sparse backend. A COO matmul is
gather-multiply-scatter, so the matrix is decomposed once into `(row, col, value)` triples
on the GPU and applied with `index_add_` — identical to `torch.sparse.mm` to 8e-7, entirely
on-device, 32 MB of temporaries against 846 MB for a dense equivalent.

`PYTORCH_ENABLE_MPS_FALLBACK` is deliberately **not** set: a genuine gap should raise, not
quietly degrade to the CPU.

## Output: configured in `config/default.yaml`

What gets stored is a config file, not code. Each variable declares its units and one or
more time aggregations:

```yaml
variables:
  2t:    {units: K,          aggregations: [native]}
  tcw:   {units: kg m**-2,   aggregations: [daily_mean]}
  tp:    {units: m,          aggregations: [daily_sum]}
```

Five aggregations are supported: **`native`** (every 6-hourly step) and **`daily_mean`**,
**`daily_min`**, **`daily_max`**, **`daily_sum`** (over the four steps of each complete UTC
calendar day, on `prediction_timedelta_daily`).

A variable may list several, which is the point of min/max — daily temperature extremes
alongside the 6-hourly field:

```yaml
2t: {units: K, aggregations: [native, daily_min, daily_max]}
#  -> 2t (6-hourly), 2t_min, 2t_max (daily)
```

**Naming rule:** a variable with one aggregation keeps its bare name, so a store built from
the shipped config is byte-compatible with the existing archive (`tcw`, not `tcw_mean`).
Only a variable producing several outputs gets suffixes, and `native` never takes one.
Every output records its `aggregation` in the variable attributes, along with `units`,
`cell_methods` and — where the name differs — `source_variable`. A source feeding several
outputs is regridded once per step, so min+max+mean of `2t` costs one regrid, not three.

Point at a different file with `--config`, restrict to some variables with `--save-fields`,
or flatten everything to 6-hourly with `--no-daily-aggregates`.

> `daily_min`/`daily_max` are extremes of the four 6-hourly **samples**, not true daily
> extremes — the model does not produce hourly output. For 2t the sampled minimum will
> usually be warmer than the real overnight low.

Encoding can also be supplied in the config, and anything omitted falls back to the code defaults
(the AIFS archive layout): chunks `(1[, 1], 24, 90, 180)`, shards `(1[, 1], 168, 720, 1440)`,
Blosc-zstd-7 bitshuffle, `BitRound(keepbits=11)` with optional per-variable overrides, and
`keepbits: null` for exact float32. `number` is present only for ensembles — deterministic
stores omit it rather than carrying a length-1 dimension.

Days are UTC calendar days: a step valid at `t` covers `(t−6h, t]`, so 00 UTC steps belong
to the previous date, and incomplete edge days are dropped.

The writer **streams**: the store is preallocated and a background thread fills up to three
region dimensions (`number`, `prediction_timedelta`, `prediction_timedelta_daily`) as steps
arrive, so peak memory tracks the write batch rather than the forecast length or ensemble
size. Stores are staged as `_partial.zarr` and renamed only on success.

Regridding uses the local CSR matrices in `support/regrid`, obtained from ECMWF's
[earthkit-geo](https://github.com/ecmwf/earthkit-geo) package and applied directly.

Open-data mirrors default to **azure → ecmwf → aws** with per-request fallback; google is
excluded because the URL layout `ecmwf-opendata` uses returns error 400.

## Reproducibility

**MPS is not bitwise deterministic.** Two identical runs differ by ~0.125 K in 2t
(deterministic) and more for ENS, whose stochastic noise amplifies it. Consequences:

- Re-running an ENS member with the same seed does *not* reproduce it bit-for-bit (~0.7 K
  vs ~10.8 K for a different seed — members are reliably distinct, but not repeatable).
  Use `--seed-offset` to extend an ensemble rather than re-running.
- Comparing two stores with `==` will always fail. Under `BitRound(keepbits=11)` the
  smallest possible difference is one quantisation step, not one ulp.

Ensemble validity is better judged by spread growth: mean σ(2t) rises monotonically from
0.29 K at +6 h to 0.48 K at +48 h.

## Tests

```bash
./aifs test
```

No network and no checkpoints required. The shared library is
version-agnostic, so the suite runs in either runtime.

They target what fails *silently* — a wrong forecast rather than a crash:

- **initial conditions** — the longitude roll, the mean-wave-direction cos/sin split, the
  soil renames, sea masking, and geopotential-height→geopotential at every level. A bug in
  any of these corrupts both models and still looks plausible on a map.
- **the shared cache** — a cached date is never re-downloaded; an interrupted write cannot
  be mistaken for a valid cache; the two models never share a forecast directory.
- **mirror fallback** — order is respected, a success short-circuits the rest, and a total
  failure names every mirror *and* its reason.
- **the MPS patches** — the stub module names resolve through the same `find_class` path
  `torch.load` uses; every stub raises rather than returning something plausible; the
  `index_add_` COO matmul equals `torch.sparse.mm` on a transposed, uncoalesced matrix.
- **the store** — the attention band against a dense masked reference and the 2³² clamp;
  the regrid operators being a partition of unity; the streaming writer against the batch
  oracle for both store shapes, every ensemble member in its own `number` slot, the daily
  accumulator resetting at member boundaries, and each of the four reducers against numpy.

## Layout

```
aifs                    dispatcher interface
aifs_mps/
  models.py             the registry — the one place the two models differ
  paths.py              asset locations, all env-overridable
  opendata.py           superset retrieval, shared cache, per-model selection
  regrid.py             0.25° ↔ N320 operators
  patches/              MPS workarounds, capability-detected
  config.py             YAML store spec: variables, units, aggregations, encoding
  zarr_layout.py        store layout + the batch path used as a test oracle
  zarr_stream.py        streaming writer (deterministic and ensemble)
  cli.py                one driver for both models
config/default.yaml     what to store and how
runtimes/{single,ens}/  version-pinning shims, each with its own uv.lock
tests/
pyproject.toml          the shared aifs-mps package; ruff and pytest config
support/                lsm.grib, regrid matrices from earthkit-geo (~21 MB)
weights/                checkpoints               (gitignored, 3.3 GB — huggingface.co/ecmwf)
input_states/           shared IC cache           (gitignored, written at runtime)
forecasts/{single,ens}/ output stores             (gitignored, written at runtime)
```
