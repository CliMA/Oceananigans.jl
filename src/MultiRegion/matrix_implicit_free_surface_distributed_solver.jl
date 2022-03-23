using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.Grids: with_halo
using Oceananigans.Architectures
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: ReducedField
using Oceananigans.Solvers: HeptadiagonalIterativeSolver
import Oceananigans.Solvers: solve!

include("unified_memory_utils.jl")

struct MatrixImplicitFreeSurfaceDistributedSolver{V, S, R}
    "The vertically-integrated lateral areas"
    vertically_integrated_lateral_areas :: V
    "The matrix iterative solver"
    matrix_iterative_solver :: S
    "The right hand side of the free surface evolution equation"
    right_hand_side :: R
end

function MatrixImplicitFreeSurfaceDistributedSolver(mrg::MultiRegionGrid, gravitational_acceleration, settings)
    
    grid = reconstruct_grid(mrg)

    # Initialize vertically integrated lateral face areas
    ∫ᶻ_Axᶠᶜᶜ = Field((Face, Center, Nothing), grid)
    ∫ᶻ_Ayᶜᶠᶜ = Field((Center, Face, Nothing), grid)

    vertically_integrated_lateral_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    compute_vertically_integrated_lateral_areas!(vertically_integrated_lateral_areas)

    right_hand_side = unified_array(zeros(eltype(grid), grid.Nx*grid.Ny))

    # Set maximum iterations to Nx * Ny if not set
    settings = Dict{Symbol, Any}(settings)
    maximum_iterations = get(settings, :maximum_iterations, grid.Nx * grid.Ny)
    settings[:maximum_iterations] = maximum_iterations

    coeffs = compute_matrix_coefficients(vertically_integrated_lateral_areas, grid, gravitational_acceleration)

    solver = HeptadiagonalIterativeSolver(coeffs; reduced_dim = (false, false, true),
                                  grid = grid, settings...)

    return MatrixImplicitFreeSurfaceSolver(vertically_integrated_lateral_areas, solver, right_hand_side)
end
