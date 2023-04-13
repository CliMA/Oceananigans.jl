using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.Grids: with_halo
using Oceananigans.Architectures
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: ReducedField
using Oceananigans.Solvers: HeptadiagonalIterativeSolver

import Oceananigans.Solvers: solve!

"""
    struct MatrixImplicitFreeSurfaceSolver{S, R, T}

The matrix-based implicit free-surface solver.

$(TYPEDFIELDS)
"""
struct MatrixImplicitFreeSurfaceSolver{S, R, T}
    "The matrix iterative solver"
    matrix_iterative_solver :: S
    "The right hand side of the free surface evolution equation"
    right_hand_side :: R
    storage :: T
end

"""
    MatrixImplicitFreeSurfaceSolver(grid::AbstractGrid, settings, gravitational_acceleration::Number)
    
Return a solver for the elliptic equation with one of the iterative solvers of IterativeSolvers.jl
with a sparse matrix formulation.
        
```math
[∇ ⋅ H ∇ - 1 / (g Δt²)] ηⁿ⁺¹ = (∇ʰ ⋅ Q★ - ηⁿ / Δt) / (g Δt) 
```
    
representing an implicit time discretization of the linear free surface evolution equation
for a fluid with variable depth `H`, horizontal areas `Az`, barotropic volume flux `Q★`, time
step `Δt`, gravitational acceleration `g`, and free surface at time-step `n` `ηⁿ`.
"""
function MatrixImplicitFreeSurfaceSolver(grid::AbstractGrid, settings, gravitational_acceleration::Number)
    
    # Initialize vertically integrated lateral face areas
    ∫ᶻ_Axᶠᶜᶜ = Field((Face, Center, Nothing), grid)
    ∫ᶻ_Ayᶜᶠᶜ = Field((Center, Face, Nothing), grid)

    vertically_integrated_lateral_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    compute_vertically_integrated_lateral_areas!(vertically_integrated_lateral_areas)

    arch = architecture(grid)
    right_hand_side = arch_array(arch, zeros(grid.Nx * grid.Ny)) # linearized RHS for matrix operations
    
    storage = deepcopy(right_hand_side)
    
    # Set maximum iterations to Nx * Ny if not set
    settings = Dict{Symbol, Any}(settings)
    maximum_iterations = get(settings, :maximum_iterations, grid.Nx * grid.Ny)
    settings[:maximum_iterations] = maximum_iterations

    coeffs = compute_matrix_coefficients(vertically_integrated_lateral_areas, grid, gravitational_acceleration)
    solver = HeptadiagonalIterativeSolver(coeffs; template = right_hand_side, reduced_dim = (false, false, true), grid, settings...)

    return MatrixImplicitFreeSurfaceSolver(solver, right_hand_side, storage)
end

build_implicit_step_solver(::Val{:HeptadiagonalIterativeSolver}, grid, settings, gravitational_acceleration) =
    MatrixImplicitFreeSurfaceSolver(grid, settings, gravitational_acceleration)

#####
##### Solve...
#####

function solve!(η, implicit_free_surface_solver::MatrixImplicitFreeSurfaceSolver, rhs, g, Δt)
    solver  = implicit_free_surface_solver.matrix_iterative_solver
    storage = implicit_free_surface_solver.storage
    
    solve!(storage, solver, rhs, Δt)
        
    set!(η, reshape(storage, solver.problem_size...))

    return nothing
end

function compute_implicit_free_surface_right_hand_side!(rhs,
                                                        implicit_solver::MatrixImplicitFreeSurfaceSolver,
                                                        g, Δt, ∫ᶻQ, η)

    solver = implicit_solver.matrix_iterative_solver
    grid = solver.grid
    arch = architecture(grid)

    event = launch!(arch, grid, :xy,
                    implicit_linearized_free_surface_right_hand_side!,
                    rhs, grid, g, Δt, ∫ᶻQ, η,
		            dependencies = device_event(arch))
    
    wait(device(arch), event)
    return nothing
end

# linearized right hand side
@kernel function implicit_linearized_free_surface_right_hand_side!(rhs, grid, g, Δt, ∫ᶻQ, η)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz + 1
    Az   = Azᶜᶜᶠ(i, j, k_top, grid)
    δ_Q  = flux_div_xyᶜᶜᶠ(i, j, k_top, grid, ∫ᶻQ.u, ∫ᶻQ.v)
    t = i + grid.Nx * (j - 1)
    @inbounds rhs[t] = (δ_Q - Az * η[i, j, k_top] / Δt) / (g * Δt)
end

function compute_matrix_coefficients(vertically_integrated_areas, grid, gravitational_acceleration)

    arch = grid.architecture

    Nx, Ny = grid.Nx, grid.Ny

    C     = arch_array(arch, zeros(eltype(grid), Nx, Ny, 1))
    diag  = arch_array(arch, zeros(eltype(grid), Nx, Ny, 1))
    Ax    = arch_array(arch, zeros(eltype(grid), Nx, Ny, 1))
    Ay    = arch_array(arch, zeros(eltype(grid), Nx, Ny, 1))
    Az    = arch_array(arch, zeros(eltype(grid), Nx, Ny, 1))

    ∫Ax = vertically_integrated_areas.xᶠᶜᶜ
    ∫Ay = vertically_integrated_areas.yᶜᶠᶜ

    event_c = launch!(arch, grid, :xy, _compute_coefficients!,
                      diag, Ax, Ay, ∫Ax, ∫Ay, grid, gravitational_acceleration,
                      dependencies = device_event(arch))
  
    wait(event_c)

    return (Ax, Ay, Az, C, diag)
end

@kernel function _compute_coefficients!(diag, Ax, Ay, ∫Ax, ∫Ay, grid, g)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Ay[i, j, 1]    = ∫Ay[i, j, 1] / Δyᶜᶠᶠ(i, j, grid.Nz+1, grid)  
        Ax[i, j, 1]    = ∫Ax[i, j, 1] / Δxᶠᶜᶠ(i, j, grid.Nz+1, grid)  
        diag[i, j, 1]  = - Azᶜᶜᶠ(i, j, grid.Nz+1, grid) / g
    end
end
