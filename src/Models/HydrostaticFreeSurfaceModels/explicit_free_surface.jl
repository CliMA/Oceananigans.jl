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

ab2_step_free_surface!(free_surface::ExplicitFreeSurface, model, Δt, χ) = 
    @apply_regionally explicit_ab2_step_free_surface!(free_surface, model, Δt, χ)

explicit_ab2_step_free_surface!(free_surface, model, Δt, χ) =
    launch!(model.architecture, model.grid, :xy,
            _explicit_ab2_step_free_surface!, free_surface.η, Δt, χ,
            model.timestepper.Gⁿ.η, model.timestepper.G⁻.η, size(model.grid, 3))

#####
##### Kernel
#####

@kernel function _explicit_ab2_step_free_surface!(η, Δt, χ, Gηⁿ, Gη⁻, Nz)
    i, j = @index(Global, NTuple)
    FT = typeof(χ)
    one_point_five = convert(FT, 1.5)
    oh_point_five = convert(FT, 0.5)
    not_euler = χ != convert(FT, -0.5)

    @inbounds begin
        Gη = (one_point_five + χ) * Gηⁿ[i, j, Nz+1] - (oh_point_five  + χ) * Gη⁻[i, j, Nz+1] * not_euler
        η[i, j, Nz+1] += Δt * Gη
    end
end

