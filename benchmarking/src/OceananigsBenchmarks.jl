module OceananigsBenchmarks

export
    # Benchmark cases
    earth_ocean,

    # Benchmark utilities
    many_time_steps!,
    benchmark_time_stepping,
    run_benchmark_simulation,
    BenchmarkResult,
    SimulationResult,
    BenchmarkMetadata

using Dates
using JLD2
using Printf
using Statistics

using Oceananigans
using Oceananigans.Architectures: GPU
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.OutputWriters: JLD2Writer, IterationInterval, TimeInterval, write_output!
using Oceananigans.Simulations: SpecifiedTimes

using NumericalEarth

using CUDA: CUDA, CUDABackend

# Base functionalities
include("metadata.jl")
include("result.jl")
include("timestepping.jl")
include("utils.jl")

# Benchmark cases
include("earth_ocean.jl")

end # module
