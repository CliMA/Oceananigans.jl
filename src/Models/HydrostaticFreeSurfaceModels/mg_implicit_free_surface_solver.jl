using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Architectures
using Oceananigans.Grids: with_halo, isrectilinear
using Oceananigans.Fields: Field, ZReducedField
using Oceananigans.Architectures: device
using Oceananigans.Models.HydrostaticFreeSurfaceModels: implicit_free_surface_linear_operation!
using Oceananigans.Solvers: fill_matrix_elements!

import Oceananigans.Solvers: solve!, precondition!
import Oceananigans.Architectures: architecture

"""
    struct MGImplicitFreeSurfaceSolver{V, S, R}

The multigrid implicit free-surface solver.

$(TYPEDFIELDS)
"""
mutable struct MGImplicitFreeSurfaceSolver{S, V, F, R}
    "The multigrid solver"
    multigrid_solver :: S
    "The vertically-integrated lateral areas"
    vertically_integrated_lateral_areas :: V
    "The previous time step"
    previous_Δt :: F
    "The right hand side of the free surface evolution equation"
    right_hand_side :: R
end

architecture(solver::MGImplicitFreeSurfaceSolver) =
    architecture(solver.multigrid_solver)

"""
    MGImplicitFreeSurfaceSolver(grid, settings)

Return a solver based on a multigrid method for the elliptic equation
    
```math
[∇ ⋅ H ∇ - 1 / (g Δt²)] ηⁿ⁺¹ = (∇ʰ ⋅ Q★ - ηⁿ / Δt) / (g Δt)
```

representing an implicit time discretization of the linear free surface evolution equation
for a fluid with variable depth `H`, horizontal areas `Az`, barotropic volume flux `Q★`, time
step `Δt`, gravitational acceleration `g`, and free surface at the `n`-th time-step `ηⁿ`.
"""
function MGImplicitFreeSurfaceSolver(grid::AbstractGrid, 
                                    settings, 
                                    gravitational_acceleration=nothing, 
                                    placeholder_timestep = -1.0)

    # Initialize vertically integrated lateral face areas
    ∫ᶻ_Axᶠᶜᶜ = Field{Face, Center, Nothing}(with_halo((3, 3, 1), grid))
    ∫ᶻ_Ayᶜᶠᶜ = Field{Center, Face, Nothing}(with_halo((3, 3, 1), grid))

    vertically_integrated_lateral_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    compute_vertically_integrated_lateral_areas!(vertically_integrated_lateral_areas)
    fill_halo_regions!(vertically_integrated_lateral_areas)
    
    # Set some defaults
    settings = Dict{Symbol, Any}(settings)
    settings[:maxiter] = get(settings, :maxiter, grid.Nx * grid.Ny)
    settings[:reltol] = get(settings, :reltol, min(1e-7, 10 * sqrt(eps(eltype(grid)))))

    right_hand_side = Field{Center, Center, Nothing}(grid)

    # initialize solver with Δt = nothing so that linear matrix is not computed; see `initialize_matrix` methods
    solver = MultigridSolver(implicit_free_surface_linear_operation!, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, gravitational_acceleration, nothing;
                             template_field = right_hand_side, settings...)

    return MGImplicitFreeSurfaceSolver(solver, vertically_integrated_lateral_areas, placeholder_timestep, right_hand_side)
end

build_implicit_step_solver(::Val{:Multigrid}, grid, settings, gravitational_acceleration) =
    MGImplicitFreeSurfaceSolver(grid, settings, gravitational_acceleration)

#####
##### Solve...
#####

function solve!(η, implicit_free_surface_solver::MGImplicitFreeSurfaceSolver, rhs, g, Δt)
    solver = implicit_free_surface_solver.multigrid_solver

    # if `Δt` changed then re-compute the matrix elements
    if Δt != implicit_free_surface_solver.previous_Δt
        ∫ᶻA = implicit_free_surface_solver.vertically_integrated_lateral_areas

        # can we get away with less re-creating_matrix below?
        fill_matrix_elements!(solver.linear_operator, η, implicit_free_surface_linear_operation!, ∫ᶻA.xᶠᶜᶜ, ∫ᶻA.yᶜᶠᶜ, g, Δt)
        implicit_free_surface_solver.previous_Δt = Δt
    end
    solve!(η, solver, rhs)

    return nothing
end


function compute_implicit_free_surface_right_hand_side!(rhs, implicit_solver::MGImplicitFreeSurfaceSolver,
                                                        g, Δt, ∫ᶻQ, η)
    solver = implicit_solver.multigrid_solver
    arch = architecture(solver)
    grid = solver.grid

    event = launch!(arch, grid, :xy,
                    implicit_free_surface_right_hand_side!,
                    rhs, grid, g, Δt, ∫ᶻQ, η,
                    dependencies = device_event(arch))
    
    wait(device(arch), event)
    return nothing
end
