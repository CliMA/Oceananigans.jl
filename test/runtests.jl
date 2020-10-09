using Test
using Logging
using Statistics
using Printf
using JLD2

using Oceananigans
using JULES

Logging.global_logger(OceananigansLogger())

@testset "JULES" begin
    include("test_models.jl")
    include("test_lazy_fields.jl")
    include("test_time_stepping.jl")
    include("test_regression.jl")
end

include("../benchmarks/benchmark_static_atmosphere.jl")
println()
