using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.Architectures
using Oceananigans.Grids: on_architecture
using Oceananigans.Fields: Field

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
             compute_vertically_integrated_lateral_areas!,
             compute_matrix_coefficients

import Oceananigans.Models.HydrostaticFreeSurfaceModels:
             build_implicit_step_solver,
             compute_implicit_free_surface_right_hand_side!

import Oceananigans.Solvers: solve!

import Oceananigans.Architectures: architecture

struct UnifiedImplicitFreeSurfaceSolver{S, R}
    unified_pcg_solver :: S
    right_hand_side :: R
end

architecture(solver::UnifiedImplicitFreeSurfaceSolver) =
    architecture(solver.preconditioned_conjugate_gradient_solver)

function UnifiedImplicitFreeSurfaceSolver(mrg::MultiRegionGrid, gravitational_acceleration::Number, settings)
    
    # Initialize vertically integrated lateral face areas
    grid = reconstruct_global_grid(mrg)
    grid = on_architecture(CPU(), grid)

    ∫ᶻ_Axᶠᶜᶜ = Field((Face, Center, Nothing), grid)
    ∫ᶻ_Ayᶜᶠᶜ = Field((Center, Face, Nothing), grid)

    vertically_integrated_lateral_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    compute_vertically_integrated_lateral_areas!(vertically_integrated_lateral_areas)

    arch = architecture(mrg) 
    right_hand_side =  unified_array(arch, zeros(eltype(grid), grid.Nx*grid.Ny))

    # Set maximum iterations to Nx * Ny if not set
    settings = Dict{Symbol, Any}(settings)
    maximum_iterations = get(settings, :maximum_iterations, grid.Nx * grid.Ny)
    settings[:maximum_iterations] = maximum_iterations

    coeffs = compute_matrix_coefficients(vertically_integrated_lateral_areas, grid, gravitational_acceleration)

    solver = UnifiedDiagonalIterativeSolver(coeffs; reduced_dim = (false, false, true),
                                            grid = grid, mrg = mrg, settings...)

                        
    return UnifiedImplicitFreeSurfaceSolver(solver, right_hand_side)
end


build_implicit_step_solver(::Val{:HeptadiagonalIterativeSolver}, grid::MultiRegionGrid, gravitational_acceleration, settings) =
    UnifiedImplicitFreeSurfaceSolver(grid, gravitational_acceleration, settings)
build_implicit_step_solver(::Val{:Default}, grid::MultiRegionGrid, gravitational_acceleration, settings) =
    UnifiedImplicitFreeSurfaceSolver(grid, gravitational_acceleration, settings)   

function compute_implicit_free_surface_right_hand_side!(rhs, implicit_solver::UnifiedImplicitFreeSurfaceSolver,
                                                        g, Δt, ∫ᶻQ, η)

    solver = implicit_solver.unified_pcg_solver
    M = length(grid.partition)
    @apply_regionally compute_regional_rhs!(rhs, solver, g, Δt, ∫ᶻQ, η, Iterate(1:M))

    return nothing
end

function compute_regional_rhs!(rhs, solver, g, Δt, ∫ᶻQ, η, region)
    
    arch = architecture(solver)
    grid = solver.grid
    event = launch!(arch, grid, :xy,
                    implicit_linearized_unified_free_surface_right_hand_side!,
                    rhs, grid, g, Δt, ∫ᶻQ, η, region * (solver.n-1),
		            dependencies = device_event(arch))

    wait(device(arch), event)
    return nothing
end

# linearized right hand side
@kernel function implicit_linearized_unified_free_surface_right_hand_side!(rhs, grid, g, Δt, ∫ᶻQ, η, displacement)
    i, j = @index(Global, NTuple)
    Az   = Azᶜᶜᶜ(i, j, 1, grid)
    δ_Q  = flux_div_xyᶜᶜᶜ(i, j, 1, grid, ∫ᶻQ.u, ∫ᶻQ.v)
    t = i + grid.Nx * (j - 1) + displacement
    @inbounds rhs[t] = (δ_Q - Az * η[i, j, 1] / Δt) / (g * Δt)
end

function solve!(η, implicit_free_surface_solver::UnifiedImplicitFreeSurfaceSolver, rhs, g, Δt)

    solver = implicit_free_surface_solver.matrix_iterative_solver
    solve!(solver, rhs, Δt)

    arch = architecture(solver)
    grid = solver.grid
    
    @apply_regionally redistribute_lhs!(η, solver.solution, arch, grid, solver.n, Iterate(1:length(solver.grid)))
    
    fill_halo_regions!(η)

    return nothing
end

function redistribute_lhs!(η, sol, arch, grid, n, region)

    event = launch!(arch, grid, :xy, _redistribute_lhs!, η, sol, region * (n-1), grid,
		            dependencies = device_event(arch))

    wait(device(arch), event)
end

# linearized right hand side
@kernel function _redistribute_lhs!(η, sol, displacement, grid)
    i, j = @index(Global, NTuple)
    t = i + grid.Nx * (j - 1) + displacement
    @inbounds η[i, j, 1] = sol[t]
end
