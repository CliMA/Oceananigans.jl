include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: materialize_advection
using Oceananigans.Utils: NormalDivision, BackendOptimizedDivision

@testset "materialize the weight computation of WENO schemes" begin
    weight_computation(::WENO{<:Any, <:Any, WCT}) where WCT = WCT
    grid = MockGrid(CPU())

    # A deferred weight computation (`Nothing`) is replaced by the default, an explicit one is kept
    deferred = WENO(Float64; order = 5, weight_computation = Nothing)
    explicit = WENO(Float64; order = 7, weight_computation = NormalDivision)

    @test weight_computation(materialize_advection(deferred, grid)) == BackendOptimizedDivision
    @test weight_computation(materialize_advection(explicit, grid)) == NormalDivision

    flux_form = materialize_advection(FluxFormAdvection(deferred, explicit, deferred), grid)
    @test weight_computation(flux_form.x) == BackendOptimizedDivision
    @test weight_computation(flux_form.y) == NormalDivision
end
