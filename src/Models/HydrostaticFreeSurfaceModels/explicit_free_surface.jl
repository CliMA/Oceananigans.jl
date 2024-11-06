using Oceananigans.Grids: AbstractGrid
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions

using Adapt

"""
    struct ExplicitFreeSurface{E, T}

The explicit free surface solver.

$(TYPEDFIELDS)
"""
struct ExplicitFreeSurface{E, G} <: AbstractFreeSurface{E, G}
    "free surface elevation"
    η :: E
    "gravitational accelerations"
    gravitational_acceleration :: G
end

ExplicitFreeSurface(; gravitational_acceleration=g_Earth) =
    ExplicitFreeSurface(nothing, gravitational_acceleration)

Adapt.adapt_structure(to, free_surface::ExplicitFreeSurface) =
    ExplicitFreeSurface(Adapt.adapt(to, free_surface.η), free_surface.gravitational_acceleration)

on_architecture(to, free_surface::ExplicitFreeSurface) =
    ExplicitFreeSurface(on_architecture(to, free_surface.η),
                        on_architecture(to, free_surface.gravitational_acceleration))

# Internal function for HydrostaticFreeSurfaceModel
function materialize_free_surface(free_surface::ExplicitFreeSurface{Nothing}, velocities, grid)
    η = free_surface_displacement_field(velocities, free_surface, grid)
    g = convert(eltype(grid), free_surface.gravitational_acceleration)

    return ExplicitFreeSurface(η, g)
end

#####
##### Kernel functions for HydrostaticFreeSurfaceModel
#####

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, free_surface::ExplicitFreeSurface) =
    free_surface.gravitational_acceleration * ∂xᶠᶜᶜ(i, j, grid.Nz+1, grid, free_surface.η)

@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface::ExplicitFreeSurface) =
    free_surface.gravitational_acceleration * ∂yᶜᶠᶜ(i, j, grid.Nz+1, grid, free_surface.η)

#####
##### Time stepping
#####

step_free_surface!(free_surface::ExplicitFreeSurface, model, timestepper, Δt) = 
    @apply_regionally explicit_ab2_step_free_surface!(free_surface, model, Δt, timestepper.χ)

explicit_ab2_step_free_surface!(free_surface, model, Δt, χ) =
    launch!(model.architecture, model.grid, :xy,
            _explicit_ab2_step_free_surface!, free_surface.η, Δt, χ,
            model.timestepper.Gⁿ.η, model.timestepper.G⁻.η, size(model.grid, 3))

#####
##### Kernel
#####

@kernel function _explicit_ab2_step_free_surface!(η, Δt, χ::FT, Gηⁿ, Gη⁻, Nz) where FT
    i, j = @index(Global, NTuple)
    @inbounds η[i, j, Nz+1] += Δt * ((FT(1.5) + χ) * Gηⁿ[i, j, Nz+1] - (FT(0.5) + χ) * Gη⁻[i, j, Nz+1])
end

# Comopute free surface tendency
function compute_free_surface_tendency!(grid, model, ::ExplicitFreeSurface, kernel_parameters)

    arch = architecture(grid)

    args = tuple(model.velocities,
                 model.free_surface,
                 model.tracers,
                 model.auxiliary_fields,
                 model.forcing,
                 model.clock)

    launch!(arch, grid, kernel_parameters,
            compute_hydrostatic_free_surface_Gη!, model.timestepper.Gⁿ.η, 
            grid, args)

    return nothing
end

#####
##### Tendency calculators for an explicit free surface
#####

""" Calculate the right-hand-side of the free surface displacement (``η``) equation. """
@kernel function compute_hydrostatic_free_surface_Gη!(Gη, grid, args)
    i, j = @index(Global, NTuple)
    @inbounds Gη[i, j, grid.Nz+1] = free_surface_tendency(i, j, grid, args...)
end
