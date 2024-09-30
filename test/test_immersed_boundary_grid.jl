include("dependencies_for_runtests.jl")

grid = RectilinearGrid(; size=(2, 2, 2), extent = (1, 1, 1))

@testset "Testing Immersed Boundaries" begin

    @info "Testing the immersed boundary construction..."

    bottom(x, y) = -1 + 0.5 * exp(-x^2 - y^2)
    ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

    # Unit test (bottom is at the right position)

    @info "Testing stably stratified initial conditions..."

end
