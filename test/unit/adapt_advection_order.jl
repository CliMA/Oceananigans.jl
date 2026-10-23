include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: adapt_advection_order

@testset "Reduced-order advection schemes have the float type of the grid" begin
    # Before, the reduced-order schemes were Float64 on every grid, which fails to compile on GPUs without Float64
    grid = RectilinearGrid(CPU(), Float32; size=(8, 8, 2), extent=(1, 1, 1))
    for scheme in (WENO(Float32; order=7), UpwindBiased(Float32; order=5), Centered(Float32; order=6))
        adapted = @test_logs (:info, r"in the z-direction because size\(grid, 3\) = 2$") adapt_advection_order(scheme, grid)
        @test adapted isa FluxFormAdvection
        @test adapted.x === scheme
        @test eltype(adapted.z) == Float32
    end
end
