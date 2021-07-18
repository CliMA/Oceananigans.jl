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

## Multithreading benchmarks

The `benchmark_multithreading.jl` script will benchmark multithreaded CPU models using incresingly more threads until it uses up all the threads on your machine. This may slow down your machine. For accurate benchmarks, there should be no computationally demanding processes running while the multithreading benchmark runs.

## Measuring performance regression

Running the `benchmark_regression.jl` script will run the incompressible model tests on the current branch and on the master branch for comparison. This is useful to test whether the current branch slows down the code or introduces any performance regression.

