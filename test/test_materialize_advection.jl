include("dependencies_for_runtests.jl")

using Oceananigans.Advection: materialize_advection
using Oceananigans.Utils: NormalDivision, ConvertingDivision, BackendOptimizedDivision



@testset "materialize weno scheme chain with placeholders" begin

    # Construct an advection chain by hand with intermediate non-WENO buffer schemes
    level_0 = Centered(order = 2)
    level_1 =
        WENO(Float64; order = 3, weight_computation = Nothing, buffer_scheme = level_0)
    level_2 = Centered(Float64; order = 4, buffer_scheme = level_1)
    level_3 = UpwindBiased(Float64; order = 5, buffer_scheme = level_2)
    level_4 =
        WENO(Float64; order = 5, weight_computation = Nothing, buffer_scheme = level_3)
    level_5 = WENO(
        Float64;
        order = 7,
        weight_computation = NormalDivision,
        buffer_scheme = level_4,
    )

    # Materialize using materialize_advection; Nothing WCTs are replaced by the global default
    # (Oceananigans.defaults.weno_weight_computation == BackendOptimizedDivision)
    materialized = materialize_advection(level_5, MockGrid(CPU()))

    # Check that all WENO schemes in the chain have the correct weight computation type
    get_nth_buffer_scheme(scheme, n) =
        n == 1 ? scheme : get_nth_buffer_scheme(scheme.buffer_scheme, n - 1)
    get_weight_computation(::WENO{<:Any,<:Any,WCT}) where {WCT} = WCT


    @test NormalDivision == get_nth_buffer_scheme(materialized, 1) |> get_weight_computation
    @test BackendOptimizedDivision ==
          get_nth_buffer_scheme(materialized, 2) |> get_weight_computation
    @test BackendOptimizedDivision ==
          get_nth_buffer_scheme(materialized, 5) |> get_weight_computation

end
