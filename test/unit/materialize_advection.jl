include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: materialize_advection
using Oceananigans.Utils: NormalDivision, BackendOptimizedDivision

@testset "materialize WENO weight computation" begin
    get_weight_computation(::WENO{<:Any, <:Any, WCT}) where {WCT} = WCT

    # A deferred weight computation takes the default (Oceananigans.defaults.weno_weight_computation == BackendOptimizedDivision)
    deferred = materialize_advection(WENO(Float64; order = 5, weight_computation = Nothing), MockGrid(CPU()))
    @test BackendOptimizedDivision == get_weight_computation(deferred)
    @test deferred.extrapolation isa TransportedWallValue

    explicit = materialize_advection(WENO(Float64; order = 7, weight_computation = NormalDivision,
                                          extrapolation = HierarchicalExtrapolation()), MockGrid(CPU()))
    @test NormalDivision == get_weight_computation(explicit)
    @test explicit.extrapolation isa HierarchicalExtrapolation
end
