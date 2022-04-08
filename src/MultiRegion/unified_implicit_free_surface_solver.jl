using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.Architectures
using Oceananigans.Grids: on_architecture
using Oceananigans.Fields: Field

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
             compute_vertically_integrated_lateral_areas!,
             compute_matrix_coefficients,
             flux_div_xyᶜᶜᶜ

import Oceananigans.Models.HydrostaticFreeSurfaceModels:
             build_implicit_step_solver,
             compute_implicit_free_surface_right_hand_side!

import Oceananigans.Solvers: solve!

import Oceananigans.Architectures: architecture

struct UnifiedImplicitFreeSurfaceSolver{A, S, R}
    vertically_integrated_areas :: A
    unified_pcg_solver :: S
    right_hand_side :: R
end

architecture(solver::UnifiedImplicitFreeSurfaceSolver) =
    architecture(solver.preconditioned_conjugate_gradient_solver)

function UnifiedImplicitFreeSurfaceSolver(mrg::MultiRegionGrid, gravitational_acceleration::Number, settings; multiple_devices = false)
    
    # Initialize vertically integrated lateral face areas
    grid = reconstruct_global_grid(mrg)
    grid = on_architecture(CPU(), grid)

    ∫ᶻ_Axᶠᶜᶜ = Field((Face, Center, Nothing), grid)
    ∫ᶻ_Ayᶜᶠᶜ = Field((Center, Face, Nothing), grid)

    vertically_integrated_lateral_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    compute_vertically_integrated_lateral_areas!(vertically_integrated_lateral_areas)
    fill_halo_regions!(vertically_integrated_lateral_areas)
    
    arch = architecture(mrg) 
    right_hand_side =  unified_array(arch, zeros(eltype(grid), grid.Nx*grid.Ny))

    # Set maximum iterations to Nx * Ny if not set
    settings = Dict{Symbol, Any}(settings)
    maximum_iterations = get(settings, :maximum_iterations, grid.Nx * grid.Ny)
    settings[:maximum_iterations] = maximum_iterations

    coeffs = compute_matrix_coefficients(vertically_integrated_lateral_areas, grid, gravitational_acceleration)

    multiple_devices ?
    solver = UnifiedDiagonalIterativeSolver(coeffs; reduced_dim = (false, false, true),
                                            grid = grid, mrg = mrg, settings...) :
    solver = HeptadiagonalIterativeSolver(coeffs; reduced_dim = (false, false, true),
                                            grid = on_architecture(arch, grid), settings...)

    return UnifiedImplicitFreeSurfaceSolver(vertically_integrated_lateral_areas, solver, right_hand_side)
end


build_implicit_step_solver(::Val{:HeptadiagonalIterativeSolver}, grid::MultiRegionGrid, gravitational_acceleration, settings) =
    UnifiedImplicitFreeSurfaceSolver(grid, gravitational_acceleration, settings)
build_implicit_step_solver(::Val{:Default}, grid::MultiRegionGrid, gravitational_acceleration, settings) =
    UnifiedImplicitFreeSurfaceSolver(grid, gravitational_acceleration, settings)   
build_implicit_step_solver(::Val{:PreconditionedConjugateGradient}, grid::MultiRegionGrid, gravitational_acceleration, settings) =
    throw(ArgumentError("Cannot use PCG solver with Multi-region grids!! Select :Default or :HeptadiagonalIterativeSolver as solver_method"))

function compute_implicit_free_surface_right_hand_side!(rhs, implicit_solver::UnifiedImplicitFreeSurfaceSolver,
                                                        g, Δt, ∫ᶻQ, η)

    grid   = ∫ᶻQ.u.grid
    M      = length(grid.partition)
    @apply_regionally compute_regional_rhs!(rhs, grid, g, Δt, ∫ᶻQ, η, Iterate(1:M), grid.partition)

    return nothing
end

function compute_regional_rhs!(rhs, grid, g, Δt, ∫ᶻQ, η, region, partition)
    arch = architecture(grid)
    event = launch!(arch, grid, :xy,
                    implicit_linearized_unified_free_surface_right_hand_side!,
                    rhs, grid, g, Δt, ∫ᶻQ, η, region, partition,
		            dependencies = device_event(arch))

    wait(device(arch), event)
    return nothing
end

# linearized right hand side
@kernel function implicit_linearized_unified_free_surface_right_hand_side!(rhs, grid, g, Δt, ∫ᶻQ, η, region, partition)
    i, j = @index(Global, NTuple)
    Az   = Azᶜᶜᶜ(i, j, 1, grid)
    δ_Q  = flux_div_xyᶜᶜᶜ(i, j, 1, grid, ∫ᶻQ.u, ∫ᶻQ.v)
    t    = displaced_xy_index(i, j, grid, region, partition)
    @inbounds rhs[t] = (δ_Q - Az * η[i, j, 1] / Δt) / (g * Δt)
end

function solve!(η, implicit_free_surface_solver::UnifiedImplicitFreeSurfaceSolver, rhs, g, Δt)

    solver = implicit_free_surface_solver.unified_pcg_solver
    
    sync_all_devices!(η.grid.devices)

    switch_device!(getdevice(solver.matrix_constructors[1]))
    sol = solve!(η, solver, rhs, Δt)

    arch = architecture(solver)
    grid = η.grid
    
    @apply_regionally redistribute_lhs!(η, sol, arch, grid, Iterate(1:length(grid)), grid.partition)

    fill_halo_regions!(η)

    return nothing
end

function redistribute_lhs!(η, sol, arch, grid, region, partition)

    event = launch!(arch, grid, :xy, _redistribute_lhs!, η, sol, region, grid, partition,
		            dependencies = device_event(arch))

    wait(device(arch), event)
end

# linearized right hand side
@kernel function _redistribute_lhs!(η, sol, region, grid, partition)
    i, j = @index(Global, NTuple)
    t = displaced_xy_index(i, j, grid, region, partition)
    @inbounds η[i, j, 1] = sol[t]
end
