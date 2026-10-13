include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: adapt_advection_order

scheme_float_type(::Centered{<:Any, FT}) where FT = FT
scheme_float_type(::UpwindBiased{<:Any, FT}) where FT = FT
scheme_float_type(::WENO{<:Any, FT}) where FT = FT

@testset "Reduced-order advection schemes have the float type of the grid" begin
    # Before, the reduced-order schemes were Float64 on every grid, which fails to compile on GPUs without Float64
    for FT in (Float32, Float64)
        grid = RectilinearGrid(CPU(), FT; size=(8, 8, 1), extent=(1, 1, 1))
        for scheme in (WENO(FT; order=5), UpwindBiased(FT; order=3), Centered(FT; order=4))
            adapted = @test_logs (:info, r"in the z-direction because size\(grid, 3\) = 1$") adapt_advection_order(scheme, grid)
            @test adapted isa FluxFormAdvection
            @test adapted.x === scheme
            @test scheme_float_type(adapted.z) == FT
        end

        grid = RectilinearGrid(CPU(), FT; size=(8, 8, 2), extent=(1, 1, 1))
        adapted = @test_logs (:info, r"in the z-direction because size\(grid, 3\) = 2$") adapt_advection_order(WENO(FT; order=7), grid)
        @test adapted.z isa WENO{2}
        @test scheme_float_type(adapted.z) == FT
    end
end
