module OceananigansBenchmarks

export
    # Benchmark cases
    earth_ocean,

    # Benchmark utilities
    many_time_steps!,
    benchmark_time_stepping,
    run_benchmark_simulation,
    run_io_benchmark,
    BenchmarkResult,
    SimulationResult,
    IOBenchmarkResult,
    BenchmarkMetadata

using Dates
using JLD2
using Printf
using Statistics

using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: sync_device!
using Oceananigans.Units
using Oceananigans.OutputWriters: write_output!

using NCDatasets

using NumericalEarth

using CUDA: CUDA

# Base functionalities
include("metadata.jl")
include("result.jl")
include("timestepping.jl")
include("utils.jl")

# Benchmark cases
include("earth_ocean.jl")

end # module
