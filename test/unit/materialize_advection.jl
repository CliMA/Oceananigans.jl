include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: materialize_advection
using Oceananigans.Utils: NormalDivision, BackendOptimizedDivision

weight_computation(::WENO{<:Any, <:Any, WCT}) where WCT = WCT

@testset "materialize WENO weight computation" begin
    grid = MockGrid(CPU())

    # `Nothing` is replaced by the default of the architecture; an explicit choice is kept
    @test weight_computation(materialize_advection(WENO(order = 5), grid)) == BackendOptimizedDivision
    @test weight_computation(materialize_advection(WENO(order = 7, weight_computation = NormalDivision), grid)) == NormalDivision
end
