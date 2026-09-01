using Oceananigans.BoundaryConditions: needs_implicit_solver
using Oceananigans.Operators: ∂xᵣᶠᶜᶠ, ∂yᵣᶜᶠᶠ

"""
$(TYPEDSIGNATURES)

Shift `velocities` by `Δt` times the barotropic acceleration `-g∇η` of `free_surface`, so that implicit momentum flux boundary 
conditions  act on a velocity that has felt the free surface. Pass `-Δt` to undo the shift once the implicit solve is done. 
The ExplicitFreeSurface, and velocities without an implicit flux boundary condition, are left alone.
"""
add_deferred_barotropic_acceleration!(velocities, grid, free_surface, Δt) = nothing

function add_deferred_barotropic_acceleration!(velocities, grid, free_surface::Union{SplitExplicitFreeSurface, ImplicitFreeSurface}, Δt)
    u, v = velocities.u, velocities.v
    needs_implicit_solver(u.boundary_conditions) | needs_implicit_solver(v.boundary_conditions) || return nothing

    g  = free_surface.gravitational_acceleration
    Δt = convert(eltype(grid), Δt)

    launch!(architecture(grid), grid, :xyz, _add_deferred_barotropic_acceleration!,
            u, v, grid, free_surface.displacement, g, Δt; exclude_periphery=true)

    return nothing
end

@kernel function _add_deferred_barotropic_acceleration!(u, v, grid, η, g, Δt)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        u[i, j, k] -= Δt * g * ∂xᵣᶠᶜᶠ(i, j, grid.Nz+1, grid, η)
        v[i, j, k] -= Δt * g * ∂yᵣᶜᶠᶠ(i, j, grid.Nz+1, grid, η)
    end
end
