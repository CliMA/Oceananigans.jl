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

using DataDeps

using CUDA: CUDA

const BATHYMETRY_URL = "https://github.com/simone-silvestri/OceananigansArtifacts.jl/raw/ss/bathymetry-for-benchmarks/bathymetry_for_benchmarks"

function __init__()
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    register(DataDep("benchmark_bathymetry",
        "Regridded bathymetry for Oceananigans benchmarks",
        ["$BATHYMETRY_URL/bathymetry_tripolar_180x90.jld2",
         "$BATHYMETRY_URL/bathymetry_tripolar_360x180.jld2",
         "$BATHYMETRY_URL/bathymetry_tripolar_720x360.jld2",
         "$BATHYMETRY_URL/bathymetry_latlon_360x180.jld2"],
        "aeeabbb81bde896b0b1c1a9d749039d564fe5e7b8870853397b9ed0ebad39667"
    ))
end

# Base functionalities
include("metadata.jl")
include("result.jl")
include("timestepping.jl")
include("utils.jl")

# Benchmark cases
include("earth_ocean.jl")

end # module
