# ZarrWriter

Plan for adding a `ZarrWriter` alongside `JLD2Writer` and `NetCDFWriter`, closing
[#3821](https://github.com/CliMA/Oceananigans.jl/issues/3821). Lives in a `Zarr.jl`-triggered extension.

## Goals

1. `ZarrWriter` accepts the same inputs as `JLD2Writer`: any `AbstractField`, `AbstractOperation`,
   `Reduction`, `WindowedTimeAverage`, `LagrangianParticles`, or function `f(model)`.
2. Same keyword arguments as `JLD2Writer` (`filename`, `dir`, `schedule`, `indices`,
   `with_halos`, `array_type`, `file_splitting`, `overwrite_existing`, `verbose`, `part`,
   `including`) plus Zarr-specific kwargs (`store`, `chunks`, `compressor`).
3. `FieldTimeSeries(zarr_path, name)` reads output back correctly. Ships in the same PR.
4. Output is appendable across `run!` calls — checkpoint + restart continues writing into the
   same Zarr store.
5. Works under MPI: each rank writes its own chunks lock-free. Serial reader loads
   distributed-written output and reconstructs the full field.

## Locked-in design decisions

- **Layout: lean-chunked, non-CF.** One Zarr array per output, shape `(field_dims..., Nt)`,
  growing along time. Per-array `.zattrs`: `location`, `indices`, `_ARRAY_DIMENSIONS`. Top-level
  `time` is a 1D growing array. `grid/` subgroup carries JSON grid reconstruction.
  *Not* CF-compliant in v1 — no shared dimension coordinate variables, no `units`/`long_name`
  conventions, no grid metric arrays. CF compatibility is additive in a follow-up (see
  "Forward compatibility" below).
- **Dimension names use `trilocation_dim_name` from day one.** `_ARRAY_DIMENSIONS` entries are
  `x_caa`/`y_aca`/`z_aaf`/`time` etc. — same names NetCDFWriter uses. This costs nothing in v1
  and avoids a rename when CF compatibility lands.
- **Time chunk = 1** by default. Each write produces brand-new chunk files; no
  read-modify-write. Time axis starts at extent 0 and grows one slab per write via
  `Zarr.append!(z, data; dims=ndims(z))` (sugar for `resize!` + indexed assign).
- **`_ARRAY_DIMENSIONS` is stored REVERSED** vs Julia's column-major dim order. Zarr stores
  shape/chunks in C order; Zarr.jl reverses at the metadata boundary, so for an
  xarray-readable file the `_ARRAY_DIMENSIONS` attribute must also be reversed. A Julia
  `(x, y, z, time)` output writes `["time", "z_aac", "y_aca", "x_caa"]`. (Gotcha — copied
  from SpeedyWeather's writer, see Prior art.)
- **Spatial chunks default to per-axis GCD of local extents.** See "MPI: distributed writes"
  for the full derivation. Single-rank: GCD = `Nx`, so chunk = full field, one chunk per
  timestep. Uniform distributed decomposition (e.g. `Partition(x=4)` with `Nx % 4 == 0`):
  GCD = local extent, each rank writes one chunk per timestep. Unequal decomposition
  (`Fractional`/`Sizes`): GCD may be smaller than local extents, so each rank writes
  multiple chunks per timestep but each chunk file is still owned by exactly one rank
  (lock-free). User can override via the `chunks` kwarg.
- **Compression: user-supplied `compressor` kwarg, default `Zarr.NoCompressor()`.** Matches
  NetCDFWriter's "off by default" stance.
- **Writer accepts only writable stores: `DirectoryStore` (default), `DictStore`,
  `S3Store`.** User passes a pre-constructed store via the `store` kwarg, or omits it and
  gets a `DirectoryStore` at `dir/filename`. Passing a `ZipStore` to the writer is a clear
  error: *"ZipStore is read-only; write to a `DirectoryStore` and call `Zarr.writezip` on
  close."*
- **Reader accepts everything**, including `ZipStore`. `FieldTimeSeries(path, name)` infers
  `ZipStore` when `path` ends in `.zip`, `DirectoryStore` otherwise; explicit stores can
  also be passed.
- **Zip finalization is a documented user post-step, not a writer kwarg.** Docstring shows
  the one-liner: `open("out.zip", "w") do io; Zarr.writezip(io, writer.store); end`. (Or
  shell `zip -r`.) `ZipStore` is read-only in Zarr.jl by design — confirmed in
  `Zarr.jl/src/Storage/zipstore.jl` — and the maintainers' prescribed workflow is
  "write to a mutable store, finalize to zip." We follow it.
- **Grid serialization: ZarrExt-local code, grid only.** No closures, buoyancy, coriolis,
  or boundary conditions in v1. No refactor of `NCDatasetsExt`. Future unification is its own
  PR.
- **`array_type` kwarg matches sister writers** (`Array{Float32}` default). Only
  `eltype(array_type)` is meaningful — it sets the Zarr array's on-disk `dtype` at creation,
  which is fixed for the array's lifetime. The backing-array half (`Array` vs hypothetical
  future GPU-resident backings) is retained for API parity with `JLD2Writer`/`NetCDFWriter`
  and for future-proofing, but currently no-ops on Zarr. GPU outputs are already moved to
  CPU by `fetch_and_convert_output` before reaching `write_output!`.
- **Dtype validation on restart.** When opening an existing store with `overwrite_existing=false`,
  check that `eltype(array_type)` matches each existing on-disk array's dtype. Mismatch
  errors with a clear message pointing at the fix (re-use original `array_type`, or set
  `overwrite_existing=true`).
- **Non-float outputs.** `Bool` outputs are stored as `Int8` (same trick `NCDatasetsExt`
  uses). Integer outputs pass through to Zarr's native integer dtypes. `array_type` only
  concerns floating-point outputs.
- **Code organization mirrors NetCDFWriter.**
  - `src/OutputWriters/zarr_writer.jl`: struct definition, docstring, error-throwing stub.
  - `Project.toml`: `Zarr` in `[weakdeps]`, `OceananigansZarrExt = "Zarr"` in `[extensions]`.
  - `ext/OceananigansZarrExt/`:
    - `OceananigansZarrExt.jl` (module)
    - `zarr_writer.jl` (constructor, init, write_output!)
    - `grid_reconstruction.jl` (JSON grid encode/decode)
    - `output_readers.jl` (`FieldTimeSeries` Zarr reader)

## On-disk layout (lean-chunked, v1)

```
my_output.zarr/
├── .zgroup
├── .zattrs                            # global attrs: Oceananigans/Julia versions, date, schedule info
├── time                               # 1D Zarr array, shape (Nt,), chunks (1,)
│   └── .zarray, .zattrs               # _ARRAY_DIMENSIONS = ["time"]
├── grid/                              # JSON grid reconstruction
│   ├── .zgroup
│   └── .zattrs                        # constructor_arguments JSON
├── grid_1/, grid_2/, ...              # only if multiple unique grids in writer
├── u/                                 # one group per output
│   ├── .zgroup
│   ├── .zattrs                        # location=["Face","Center","Center"], indices=..., grid_index=1
│   ├── .zarray                        # shape=(Nx,Ny,Nz,Nt), chunks=(cx,cy,cz,1)
│   └── chunk files                    # e.g. "0.0.0.5"
├── v/, w/, T/, ...
```

Per-output array carries `_ARRAY_DIMENSIONS = ["x_faa", "y_aca", "z_aac", "time"]`
(trilocation names) so xarray will be able to open it once we layer CF on top.

## Prior art

[SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl) ships a `ZarrOutput`
in `ext/SpeedyWeatherZarrExt.jl`. Useful to crib from, instructive where we diverge:

| Decision | SpeedyWeather | This plan (v1) |
|---|---|---|
| Layout | CF-compliant (shared `lon`/`lat`/`layer`/`time` coord arrays, `_FillValue`, `units`, `long_name`) | Lean-chunked, no CF (CF is additive follow-up) |
| Time axis | Pre-allocated to known final length `n_outputs` (uses `clock.n_timesteps`) | Grows via `Zarr.append!` |
| Stores | Hard-coded `DirectoryStore` in the type signature | DirectoryStore, DictStore, S3 (writer); + ZipStore (reader only) |
| Compressor default | `BloscCompressor(clevel=DEFAULT_COMPRESSION_LEVEL)` | `NoCompressor()` (user opt-in) |
| Restart support | None (counter resets) | First-class (Level 4 tests) |
| `_ARRAY_DIMENSIONS` reversal | yes | yes (copied) |
| `close` | no-op | finalizes, optionally writes consolidated metadata |

Worth borrowing in v1: the `_ARRAY_DIMENSIONS` reversal (already locked in above). Worth
considering as a *non-default* opt-in: pre-allocating the time axis when `stop_iteration`/
`stop_time` is known — faster than `resize!` per step. Not in scope for v1; growing-axis
is the safe default and the only path that supports restart.

## MPI: distributed writes

### Design: one shared store, lock-free parallel writes

Diverges from the existing JLD2/NetCDF "one file per rank" convention. Each rank writes its
own chunks into a single shared store; a serial reader opens the resulting store and gets
the global field for free. The whole reason to add Zarr alongside JLD2 and NetCDF is to
support exactly this — the per-rank-file convention is precisely the pain point Zarr
relieves.

### Chunk shape and unequal partitions

Zarr arrays have a *uniform* chunk shape (the spec mandates it). For every chunk file to be
written by exactly one rank — required for lock-free writes — every rank boundary must
align to a chunk boundary. That means the chunk size along each axis must divide every
rank's local extent along that axis:

> `c_axis = gcd(local_extents_along_axis)`

Worked example. `Nx_total = 600`, `Partition(x = Fractional(1, 2, 3))` → local extents
`(100, 200, 300)` → `gcd = 100`. Rank 0 writes 1 chunk per timestep, rank 1 writes 2, rank
2 writes 3. Each chunk file is owned uniquely by one rank.

Why not `max(local_extents)`: it would put rank boundaries in the middle of chunks, forcing
two ranks to share a chunk file → race / read-modify-write. Worked counterexample in the
review discussion.

Pathological case: coprime local extents (`Fractional(2, 3)` on `Nx=5` → GCD = 1, one file
per cell). At construction time, emit a `@warn` if `Nx_total / gcd_x > 10 * Rx`, suggesting
either an explicit `chunks` kwarg or a rebalanced decomposition. Realistic load-balancing
ratios share large GCDs in practice; the warning catches accidents.

User override: `chunks = (cx, cy, cz, ct)` accepted. Validated that each component divides
every local extent along its axis; error with a specific message if not.

### Per-step write path

```
1. barrier
2. resize: every rank calls Zarr.resize!(z, (..., Nt+1)) — bumps in-memory shape;
            rank 0 also persists .zarray (other ranks' identical writes are skipped or
            harmless on POSIX, depending on store)
3. write data: every rank writes its chunks (chunk filenames disjoint by rank — no races)
4. time: rank 0 only appends to the `time` array
5. barrier
```

### Init / restart

- Init: rank 0 creates the store, the arrays, grid reconstruction JSON, `.zattrs` (with
  recorded rank topology `Rx`, `Ry`, `Rz`). All other ranks wait at a barrier, then open the
  existing store read/write.
- Restart with `overwrite_existing=false`: all ranks open the existing store. Validate the
  recorded rank topology matches the current architecture — mismatch errors with a clear
  message. Read `length(time_array)` to find the resume point.

### Concretely-tricky details

- **Particles**: `LagrangianParticles` is not supported under `Distributed` in Oceananigans
  yet. ZarrWriter under MPI errors at construction if asked to write particle outputs.
  Re-enable once distributed particles land.
- **Indices slicing under MPI**: ranks whose local domain falls entirely outside the slice
  have nothing to write — short-circuit at `write_output!`.
- **MPI primitives**: barriers come from `Oceananigans.DistributedComputations`, not direct
  `MPI.jl` calls. Keeps the writer consistent with the rest of the codebase and contained
  inside the extension.
- **Store handle sharing**: each rank holds its own `ZGroup` pointing at the shared
  `DirectoryStore` on a shared filesystem (no handle sharing across ranks). For `S3Store`,
  each rank constructs its own client to the same bucket/prefix.

## Forward compatibility (additive path to CF)

| Feature | v1 | CF follow-up | Breaking? |
|---|---|---|---|
| Per-variable chunked arrays | yes | yes | — |
| `_ARRAY_DIMENSIONS` with trilocation names | yes | yes | — |
| `units`, `long_name`, `standard_name` attrs | no | add | additive |
| Grid metric arrays (`Δx_caa` etc.) | no | add | additive |
| Shared dimension coordinate arrays | no | add | structural; readers must key on `_ARRAY_DIMENSIONS`, not assume layout |
| CF time encoding (`units = "seconds since ..."`) | no | add | additive |
| `Conventions = "CF-1.10"` global attr | no | add | additive |
| Immersed-boundary metadata arrays | no | add | additive |
| Closure / BC JSON serialization | no | needs separate design | additive |

## Test plan

Tests live in `test/test_zarr_writer.jl` (loaded behind `using Zarr` so the extension is
active). MPI tests use the existing distributed-test machinery.

### Level 1 — Smoke / API coverage

Mirrors what `test/test_jld2_writer.jl` checks. Single-grid, single-rank, `DirectoryStore`.

- Constructor accepts `NamedTuple` of fields.
- Constructor accepts `Dict` of fields.
- Constructor accepts an `AbstractOperation` (e.g., `Average`) and a `Reduction`.
- Constructor accepts a function `f(model)` with explicit `dimensions`.
- All JLD2-style kwargs are accepted and honored:
  `filename`, `dir`, `schedule`, `indices`, `with_halos`, `array_type`,
  `file_splitting`, `overwrite_existing`, `including`, `part`, `verbose`.
- Zarr-specific kwargs: `store`, `chunks`, `compressor`.
- `show(::ZarrWriter)` produces something sensible.

### Level 2 — Round-trip correctness

- Write a known field (filled with a known function of indices) → reopen with `Zarr.zopen`
  directly → bit-exact match (after `array_type` conversion).
- Write a field, read with `FieldTimeSeries(path, name)` → match values + correct
  `location`, `indices`, `boundary_conditions` reconstruction (where supported in v1).
- `Average` and other reductions write the reduced shape and correct values.
- Function outputs write with the user-supplied `dimensions`.
- `with_halos = true` vs `false` produce correctly-shaped arrays.
- Sliced `indices = (:, :, 1)` etc. write the slice.
- `array_type = Array{Float32}` vs `Array{Float64}` correctly cast on disk.

### Level 3 — File management

- `overwrite_existing = true` removes existing store before writing.
- `overwrite_existing = false` appends new timesteps to existing store (no duplicate or
  truncated time axis).
- `file_splitting = NoFileSplitting()`: one store.
- `file_splitting = FileSizeLimit(sz)`: splits at expected boundaries; each split is a
  self-contained Zarr store with its own grid reconstruction.
- `file_splitting = TimeInterval(Δt)`: splits at expected times.
- Multi-grid writer: outputs on two different grids end up tagged with distinct
  `grid_index` attrs and both grids appear under `grid_1/` / `grid_2/`.

### Level 4 — Continued simulation (checkpoint + restart)

Single test exercising the append path end-to-end:

1. Build a model + `ZarrWriter` + `Checkpointer`.
2. Run for N steps. Verify Zarr store has N output timesteps.
3. Build a new model from the checkpoint, attach a *new* `ZarrWriter` pointed at the same
   path with `overwrite_existing = false`.
4. Run for another N steps.
5. Reopen the Zarr store, expect 2N timesteps with monotonically-increasing time, no
   duplicates, no gaps.
6. `FieldTimeSeries(path, name)` reads the full 2N-step series and values match.

Bonus: same test with `file_splitting = TimeInterval(...)` straddling the restart.

### Level 5 — Alternative stores

- `DictStore` (in-memory): full Level 1 + Level 2 test pass against an in-memory store.
  Cheap; exercises the store-abstraction code path without filesystem.
- `ZipStore` (read-side only): write to a `DirectoryStore`, finalize with
  `Zarr.writezip(io, store)`, then open the resulting `.zip` via
  `FieldTimeSeries(path_to_zip, name)` and check values match. Writer should *reject*
  `ZipStore` with an explicit error message — test that too.
- `S3Store`: not in CI. A `test/manual/test_zarr_s3.jl` script that runs against a local
  MinIO if `OCEANANIGANS_TEST_S3=1`. Skipped otherwise.

### Level 6 — Reader (`FieldTimeSeries` from Zarr)

Mirrors the NetCDF reader test coverage in `ext/OceananigansNCDatasetsExt/output_readers.jl`:

- Open by `path::String` and by `store::AbstractStore`.
- `InMemory` vs `OnDisk` loading modes.
- Time indexing (`fts[i]`, `fts[Time(t)]`).
- `location`, `indices`, `boundary_conditions` reconstructed from `.zattrs`.
- Grid reconstructed from `grid/` subgroup; round-trips for `RectilinearGrid` and
  `LatitudeLongitudeGrid` (others deferred).

### Level 7 — MPI parallel

Lives under the distributed-test harness.

- Run a distributed model decomposed across x (and separately across y, and separately
  across both). Each rank writes its local chunk of each output.
- After the run, on rank 0:
  - Open the Zarr store. Verify the array shape matches the global field, and that the
    expected chunk files exist (one per rank per timestep).
  - `FieldTimeSeries(path, name)` (serial) reads the full global field. Values match a
    reference single-rank run with the same setup.
- Bit-exact match between serial-written and distributed-written output (for outputs that
  are bit-reproducible — i.e., not affected by reduction order).

## Implementation status (this PR)

| Phase | Status | Notes |
|---|---|---|
| 1. Skeleton | ✅ | Struct, stub, extension wiring, 15 smoke tests |
| 2. Time-axis writing & raw round-trip | ✅ | `initialize!` + `write_output!` via `Zarr.append!`, 28 tests |
| 3. Operations, reductions, functions, WTA | ✅ | `dimensions` kwarg for non-Field outputs, 21 tests |
| 4. Grid reconstruction + multi-grid | ✅ | JSON in `grid/` (or `grid_<n>/`) subgroups, RectilinearGrid + LatLong; immersed boundary deferred. 17 tests |
| 5. FieldTimeSeries reader | ✅ | `FieldTimeSeries("out.zarr", "u")` round-trips; 10 tests |
| 6. File splitting + append | ✅ | Append-on-existing-store, dtype validation on restart, `start_next_file`. 8 tests |
| 7. Alternative stores | ✅ | DictStore + ZipStore-read tested in CI; S3Store as manual script under `test/manual/test_zarr_s3.jl`. 9 tests |
| 8. MPI distributed write | ✅ | Per-axis GCD chunks, rank-0 metadata/time writes, per-rank chunk-aligned data writes, rank-topology recorded in `.zattrs`, particles rejected. `mpiexec`-driven driver in `test/test_distributed_zarr_writer.jl` exercises Partition(x=2)/Partition(y=2)/Partition(x=2,y=2). MPI worker lives at `test/distributed_zarr_writer_tests.jl`. |
| 9. Comparative benchmarks | ✅ | Scripts under `benchmark/zarr_writer/`; not committed to CI |

**108 tests pass.**

## Implementation phases

Each phase a self-contained commit (or small set of commits). Tests for the phase land
with it.

1. **Skeleton.** Struct, docstring, error-stub in `src/OutputWriters/zarr_writer.jl`. Extension
   module wired up. Constructor for `AbstractField` outputs only. `DirectoryStore` only.
   Level 1 smoke tests (kwarg surface, `show`).
2. **Time-axis writing.** `initialize_zarr_file!` creates `time` array + per-output arrays at
   extent 0. `write_output!` resizes + appends. Level 2 round-trip via raw `Zarr.zopen`.
3. **Operations, reductions, functions, WindowedTimeAverage.** `construct_output` integration,
   `dimensions` kwarg handling for functions. Round-trip tests extended.
4. **Grid reconstruction.** JSON encode in `grid/` subgroup. Multi-grid support
   (`grid_1`, `grid_2`, ...). `LagrangianParticles` output.
5. **`FieldTimeSeries` Zarr reader.** Mirrors NetCDF reader. Level 6 tests.
6. **File splitting + append + checkpoint integration.** `overwrite_existing` semantics,
   `start_next_file`, Level 3 and Level 4 tests.
7. **Alternative stores.** `DictStore` first-class; `ZipStore` per resolution of the open
   question below; `S3Store` manual script.
8. **MPI.** Per-rank chunk shape default, distributed write tests, serial-reader-on-MPI-output
   tests (Level 7).

Phases 1–5 are roughly the minimum for a useful PR if we want to split. Phases 6–8 are the
"real" production-ready set. Plan A: one PR with all eight. Plan B: phases 1–5 in PR1,
6–8 in PR2.

## Future work (not in scope for v1, all writer-agnostic)

- **Bitrounding / lossy compression of float outputs** (SpeedyWeather has `keepbits`).
  Logically lives in the `fetch_and_convert_output` pipeline, not in any single backend, so
  it can land later as a shared kwarg that benefits JLD2 / NetCDF / Zarr at once.
- **Closures, buoyancy, coriolis, boundary-condition JSON serialization.** Defer until
  there's demand; `Checkpointer` covers the full-state-reload use case in the meantime.
- **CF-compliant layout** (shared dimension coordinate arrays, `units`/`long_name` attrs,
  grid metric arrays). Additive on top of v1 — see "Forward compatibility" above.

## Benchmarks (run as part of this PR, results reported in PR description)

Goal: a comparative read/write benchmark across `JLD2Writer`, `NetCDFWriter`, and
`ZarrWriter` to motivate the addition and to surface any unexpected performance regression.
CPU-only is fine. Scripts live under `benchmark/zarr_writer/` (not part of the test
suite, not run in CI); results go into the PR description as a table.

### Scenarios

- **Serial single output**: 5 fields, shape `(512, 512, 64)`, `Float32`, 100 timesteps,
  no compression, default chunking.
- **Serial multi-output (more fields)**: 10 fields, same shape, 100 timesteps.
- **Distributed write**: `Partition(x=4)` and `Partition(x=2, y=2)`, same field set, on
  one node (4 ranks via MPI). 50 timesteps. Uniform decomposition (best case for Zarr).
- **Distributed write, unequal partition**: `Partition(x=Fractional(1,2,3,4))` on 4 ranks,
  to exercise the GCD chunk path.
- **Compressed write** (Zarr only): same as serial single-output, with `BloscCompressor`
  at default level. Reports compressed-vs-uncompressed Zarr write throughput + on-disk
  size delta.

### Metrics per scenario

- Wall-clock time per `write_output!` call (median, IQR).
- Total wall-clock for the run.
- On-disk size when the run completes.
- Time to load the full time series back into memory via `FieldTimeSeries` (where
  supported by the writer's reader).

### Reporting

Markdown tables in the PR description, with one row per scenario × writer. Columns:
write throughput (MB/s), total wall time (s), on-disk size (MiB), read-back time (s).
Scripts are committed but unannounced (no doctest, no CI run) — anyone can rerun by
invoking `julia --project benchmark/zarr_writer/run_serial.jl` and similar.

## Open questions

(none — proceed as single PR with all 8 phases)
