# ZarrWriter comparative benchmarks

Comparing `JLD2Writer`, `NetCDFWriter`, and `ZarrWriter` on CPU. Not part of CI —
run locally and paste the resulting markdown tables into the PR description.

## Setup

From the repo root:

```bash
cd benchmark/zarr_writer
julia --project=. -e 'using Pkg; Pkg.develop(path="../.."); Pkg.instantiate()'
```

## Scripts

Each script prints a markdown table to stdout and (optionally) writes the same to a
CSV file. They all run on CPU only.

| Script | Scenario |
|---|---|
| `serial_single.jl` | 5 fields, `(128, 128, 64)` `Float32`, 50 timesteps, no compression |
| `serial_multi.jl`  | 10 fields, same shape and step count |
| `compressed.jl`    | Same as `serial_single.jl` but `ZarrWriter` uses `BloscCompressor(clevel=3)` |

Distributed (MPI) scenarios are deferred — ZarrWriter's MPI path is in a follow-up PR.

Example:

```bash
julia --project=. serial_single.jl
```

Metrics reported per (writer, scenario):

- Total wall time for the run (seconds)
- Mean per-step write latency (ms)
- On-disk size at end of run (MiB)
- Read-back time for the full timeseries via `FieldTimeSeries` (seconds)
