using Test

using Oceananigans
using JULES

@testset "JULES" begin
    include("test_models.jl")
    include("test_time_stepping.jl")
end

include("../benchmarks/benchmark_static_atmosphere.jl")
println()
