include("dependencies_for_runtests.jl")

grid = RectilinearGrid(; size=(100, 100, 100), extent = (1, 1, 1))

@testset "Testing Immersed Boundaries" begin

    @info "Testing the immersed boundary construction..."
    bottom(x, y) = rand()

    ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

    # Test that the bottom is at the same position
    bottom_height = interior(ibg.immersed_boundary.bottom_height)

    zfaces = znodes(ibg, Face())

    for i in 1:size(ibg, 1), j in 1:size(ibg, 2)
        @test bottom_height[i, j, 1] ∈ zfaces
    end
end
