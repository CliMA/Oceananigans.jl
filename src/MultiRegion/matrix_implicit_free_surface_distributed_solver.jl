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
    
    # Initialize vertically integrated lateral face areas
    ∫ᶻ_Axᶠᶜᶜ = Field((Face, Center, Nothing), mrg)
    ∫ᶻ_Ayᶜᶠᶜ = Field((Center, Face, Nothing), mrg)

    vertically_integrated_lateral_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    @apply_regionally compute_vertically_integrated_lateral_areas!(vertically_integrated_lateral_areas)
    fill_halo_regions!(vertically_integrated_lateral_areas)
    
    arch = architecture(mrg)

    dims = total_length(mrg, mrg.partition)

    right_hand_side = unified_zeros(eltype(mrg), arch, dims[1] * dims[2])

    # Set maximum iterations to Nx * Ny if not set
    settings = Dict{Symbol, Any}(settings)
    maximum_iterations = get(settings, :maximum_iterations, dims[1] * dims[2])
    settings[:maximum_iterations] = maximum_iterations

    coeffs = construct_regionally(compute_matrix_coefficients, vertically_integrated_lateral_areas, grid, gravitational_acceleration)

    solver = construct_regionally(HeptadiagonalIterativeSolver, coeffs; reduced_dim = (false, false, true),
                                  grid = mrg, settings...)

    return MatrixImplicitFreeSurfaceSolver(vertically_integrated_lateral_areas, solver, right_hand_side)
end
