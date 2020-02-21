using Test

using Oceananigans
using JULES

using JULES: VaporPlaceholder, VaporLiquidIcePlaceholder

@testset "JULES" begin
    include("test_models.jl")
    include("test_time_stepping.jl")
end
