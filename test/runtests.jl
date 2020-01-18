using Test

@testset "JULES" begin
    include("test_models.jl")
    include("test_time_stepping.jl")
end
