using .SplitExplicitFreeSurfaces: barotropic_split_explicit_corrector!
using Oceananigans.Operators: ∂xᶠᶜᶠ, ∂yᶜᶠᶠ

#####
##### Barotropic pressure correction for models with a free surface
#####

"""
    correct_barotropic_mode!(model::HydrostaticFreeSurfaceModel, Δt; kwargs...)

Correct baroclinic velocities to be consistent with the updated barotropic mode.

The correction depends on the free surface type:
- `ImplicitFreeSurface`: Subtracts the barotropic pressure gradient from velocities
- `SplitExplicitFreeSurface`: Reconciles baroclinic and barotropic velocity modes
- `ExplicitFreeSurface` / `Nothing`: No correction needed
"""
correct_barotropic_mode!(model::HydrostaticFreeSurfaceModel, Δt; kwargs...) =
    correct_barotropic_mode!(model, model.free_surface, Δt; kwargs...)

# Fallback for ExplicitFreeSurface and Nothing free surfaces
correct_barotropic_mode!(model, ::Nothing, Δt; kwargs...) = nothing
correct_barotropic_mode!(model, ::ExplicitFreeSurface, Δt; kwargs...) = nothing

#####
##### Barotropic pressure correction for models with an Implicit free surface
#####

"""
    correct_barotropic_mode!(model, ::ImplicitFreeSurface, Δt)

Apply barotropic pressure correction for implicit free surface.
After solving the implicit free surface equation, velocities are corrected by
adding the barotropic pressure gradient: `u -= g * Δt * ∂η/∂x`, `v -= g * Δt * ∂η/∂y`.
"""
function correct_barotropic_mode!(model, ::ImplicitFreeSurface, Δt)
    refresh_prescribed_velocity_state!(model, model.velocities)

    launch!(model.architecture, model.grid, volume_kernel_parameters(model.grid),
            _barotropic_pressure_correction!,
            model.velocities,
            model.grid,
            Δt,
            model.free_surface.gravitational_acceleration,
            model.free_surface.displacement)

    return nothing
end

"""
    correct_barotropic_mode!(model, ::SplitExplicitFreeSurface, Δt)

Reconcile baroclinic and barotropic velocity modes for split-explicit free surface.
The depth-averaged baroclinic velocity is corrected to match the barotropic velocity
from the split-explicit substepping.
"""
function correct_barotropic_mode!(model, ::SplitExplicitFreeSurface, Δt)
    refresh_prescribed_velocity_state!(model, model.velocities)
    u, v, _ = model.velocities
    grid = model.grid
    barotropic_split_explicit_corrector!(u, v, model.free_surface, grid)

    return nothing
end

@inline implicit_free_surface_barotropic_pressure_gradient_u(i, j, k, grid, η) =
    ∂xᶠᶜᶠ(i, j, grid.Nz+1, grid, η)

@inline implicit_free_surface_barotropic_pressure_gradient_v(i, j, k, grid, η) =
    ∂yᶜᶠᶠ(i, j, grid.Nz+1, grid, η)

@kernel function _barotropic_pressure_correction!(U, grid, Δt, g, η)
    i, j, k = @index(Global, NTuple)

    u = Oceananigans.Advection.u_velocity(U)
    v = Oceananigans.Advection.v_velocity(U)

    @inbounds begin
        u[i, j, k] -= g * Δt * implicit_free_surface_barotropic_pressure_gradient_u(i, j, k, grid, η)
        v[i, j, k] -= g * Δt * implicit_free_surface_barotropic_pressure_gradient_v(i, j, k, grid, η)
    end
end
