# GPU Benchmark Plan

All benchmarks use: Float32, SplitRungeKutta3, Nz=100, dt=60s, 100 time steps, 10 warmup steps.
Default configuration: tripolar grid with bathymetry, CATKE closure,
WENO9 momentum / WENO7 tracer advection, T+S tracers.

Results are appended to `benchmark_results.json` and auto-summarized in `benchmark_results.md`.

## Group 1 — Resolution sweep

Tripolar + bathymetry, CATKE, WENO9/WENO7, T+S.

| Label | Size          |
|-------|---------------|
| 1deg  | 360x180x100   |
| 1/2deg| 720x360x100   |
| 1/4deg| 1440x720x100  |

```bash
julia --project run_benchmarks.jl --size=360x180x100 --output=benchmark_results.json --clear
julia --project run_benchmarks.jl --size=720x360x100 --output=benchmark_results.json
julia --project run_benchmarks.jl --size=1440x720x100 --output=benchmark_results.json
```

## Group 2 — Closure sweep (1/4 deg, tripolar + bathymetry)

WENO9/WENO7, T+S.

| Closure              |
|----------------------|
| nothing              |
| CATKE                |
| CATKE+Biharmonic     |
| CATKE+GM+Biharmonic  |

```bash
julia --project run_benchmarks.jl --size=1440x720x100 --closure=nothing --output=benchmark_results.json
julia --project run_benchmarks.jl --size=1440x720x100 --closure=CATKE --output=benchmark_results.json
julia --project run_benchmarks.jl --size=1440x720x100 --closure=CATKE+Biharmonic --output=benchmark_results.json
julia --project run_benchmarks.jl --size=1440x720x100 --closure=CATKE+GM+Biharmonic --output=benchmark_results.json
```

## Group 3 — Advection sweep (1/4 deg, tripolar + bathymetry)

CATKE, T+S.

| Momentum               | Tracer  |
|-------------------------|---------|
| nothing                 | nothing |
| WENOVectorInvariant5    | WENO5   |
| WENOVectorInvariant9    | WENO9   |

```bash
julia --project run_benchmarks.jl --size=1440x720x100 --momentum_advection=nothing --tracer_advection=nothing --output=benchmark_results.json
julia --project run_benchmarks.jl --size=1440x720x100 --momentum_advection=WENOVectorInvariant5 --tracer_advection=WENO5 --output=benchmark_results.json
julia --project run_benchmarks.jl --size=1440x720x100 --momentum_advection=WENOVectorInvariant9 --tracer_advection=WENO9 --output=benchmark_results.json
```

## Group 4 — Grid type sweep (1/4 deg)

CATKE, WENO9/WENO7, T+S.

| Grid                                  |
|---------------------------------------|
| lat_lon_flat (no bathymetry, no IBM)  |
| lat_lon (+ earth bathymetry + IBM)    |
| tripolar (+ earth bathymetry + IBM)   |

```bash
julia --project run_benchmarks.jl --size=1440x720x100 --grid_type=lat_lon_flat --output=benchmark_results.json
julia --project run_benchmarks.jl --size=1440x720x100 --grid_type=lat_lon --output=benchmark_results.json
julia --project run_benchmarks.jl --size=1440x720x100 --grid_type=tripolar --output=benchmark_results.json
```

## Group 5 — Tracer count sweep (1/4 deg, tripolar + bathymetry)

CATKE, WENO9/WENO7.

| Tracers                             |
|-------------------------------------|
| T,S,C1 (3)                         |
| T,S,C1,C2,C3 (5)                   |
| T,S,C1,C2,C3,C4,C5,C6,C7,C8 (10)  |

```bash
julia --project run_benchmarks.jl --size=1440x720x100 --tracers=T,S,C1 --output=benchmark_results.json
julia --project run_benchmarks.jl --size=1440x720x100 --tracers=T,S,C1,C2,C3 --output=benchmark_results.json
julia --project run_benchmarks.jl --size=1440x720x100 --tracers=T,S,C1,C2,C3,C4,C5,C6,C7,C8 --output=benchmark_results.json
```

## Group 6 — IO benchmark (1/4 deg, tripolar + bathymetry)

CATKE, WENO9/WENO7, T+S, 1440 time steps.

| Format | Output interval |
|--------|-----------------|
| jld2   | 1, 6, 144       |
| netcdf | 1, 6, 144       |

```bash
julia --project run_benchmarks.jl --mode=io --size=1440x720x100 --time_steps=1440 --output_format=jld2 --output_iteration_interval=1 --output=benchmark_results.json
julia --project run_benchmarks.jl --mode=io --size=1440x720x100 --time_steps=1440 --output_format=jld2 --output_iteration_interval=6 --output=benchmark_results.json
julia --project run_benchmarks.jl --mode=io --size=1440x720x100 --time_steps=1440 --output_format=jld2 --output_iteration_interval=144 --output=benchmark_results.json
julia --project run_benchmarks.jl --mode=io --size=1440x720x100 --time_steps=1440 --output_format=netcdf --output_iteration_interval=1 --output=benchmark_results.json
julia --project run_benchmarks.jl --mode=io --size=1440x720x100 --time_steps=1440 --output_format=netcdf --output_iteration_interval=6 --output=benchmark_results.json
julia --project run_benchmarks.jl --mode=io --size=1440x720x100 --time_steps=1440 --output_format=netcdf --output_iteration_interval=144 --output=benchmark_results.json
```

## Quick CPU smoke test

```bash
julia --project run_benchmarks.jl \
  --device=CPU --size=90x45x10 --time_steps=10 \
  --closure=CATKE+GM+Biharmonic --grid_type=lat_lon_flat --tracers=T,S,C1,C2,C3
```
