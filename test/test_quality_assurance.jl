using Oceananigans: Oceananigans
using Aqua: Aqua
using Test: @testset

@testset "Aqua" begin
    Aqua.test_all(Oceananigans; ambiguities=false, undefined_exports=false)
end
