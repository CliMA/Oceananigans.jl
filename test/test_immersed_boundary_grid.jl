include("dependencies_for_runtests.jl")

@testset "Testing Immersed Boundaries" begin
    grid = RectilinearGrid(; size=(100, 100, 100), extent = (1, 1, 1))
    bottom_boundaries = (GridFittedBottom, PartialCellBottom)

    @info "Testing the immersed boundary construction..."
    bottom(x, y) = rand()

    for bottom_boundary in bottom_boundaries
        ibg = ImmersedBoundaryGrid(grid, bottom_boundary(bottom))
        @test summary(ibg) isa String
    end

    ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

    # Test that the bottom is at the same position
    bottom_height = interior(ibg.immersed_boundary.bottom_height)
    zfaces = znodes(ibg, Face())

    for i in 1:size(ibg, 1), j in 1:size(ibg, 2)
        @test bottom_height[i, j, 1] ∈ zfaces
    end

    # Test immersed dot product
    underlying_grid = RectilinearGrid(size = (10, 10, 10), extent = (1, 1, 1))  # grid with 1000 points
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(-0.5))        # of which only 500 are active
    c = CenterField(grid)
    fill!(c, 1)

    @test dot(c, c) == 500
end
