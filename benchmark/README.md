# Oceananigans.jl performance benchmarks

This directory contains scripts and modules for benchmarking various features of Oceananigans.

To instantiate the benchmarks environment, run

```
julia -e 'using Pkg; Pkg.activate(pwd()); Pkg.instantiate(); Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))'
```

Once the environment has been instantiated, benchmarks can be run via, e.g.

```
julia --project benchmark_nonhydrostatic_model.jl
```

Most scripts benchmark one feature (e.g. advection schemes, arbitrary tracers). If your machine contains a CUDA-compatible GPU, benchmarks will also run on the GPU. Tables with benchmark results will be printed (and each table will also be saved to an HTML file).

## Distributed benchmarks

Run distributed benchmarks by running the launcher scripts for either the shallow water model: `distributed_shallow_water_model.jl` or the nonhydrostatic model: `distributed_nonhydrostatic_model.jl`. Change settings within the scripts to toggle between strong or weak scaling and threaded or MPI architecture. The single and serial scripts executed by the launcher scripts can also be executed manually from the command line with the appropriate arguments.

## Measuring performance regression

Running the `benchmark_regression.jl` script will run the nonhydrostatic model tests on the current branch and on the master branch for comparison. This is useful to test whether the current branch slows down the code or introduces any performance regression.

