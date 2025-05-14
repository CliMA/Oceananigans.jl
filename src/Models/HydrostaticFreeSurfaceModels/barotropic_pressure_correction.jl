using .SplitExplicitFreeSurfaces: barotropic_split_explicit_corrector!
import Oceananigans.TimeSteppers: calculate_pressure_correction!, pressure_correct_velocities!

calculate_pressure_correction!(::HydrostaticFreeSurfaceModel, Δt) = nothing

#####
##### Barotropic pressure correction for models with a free surface
#####

pressure_correct_velocities!(model::HydrostaticFreeSurfaceModel, Δt; kwargs...) =
    pressure_correct_velocities!(model, model.free_surface, Δt; kwargs...)

# Fallback
pressure_correct_velocities!(model, free_surface, Δt; kwargs...) = nothing

#####
##### Barotropic pressure correction for models with an Implicit free surface
#####

function pressure_correct_velocities!(model, ::ImplicitFreeSurface, Δt)

    launch!(model.architecture, model.grid, :xyz,
            _barotropic_pressure_correction!,
            model.velocities,
            model.grid,
            Δt,
            model.free_surface.gravitational_acceleration,
            model.free_surface.η)

    return nothing
end

function pressure_correct_velocities!(model, ::SplitExplicitFreeSurface, Δt)
    u, v, _ = model.velocities
    grid = model.grid
    barotropic_split_explicit_corrector!(u, v, model.free_surface, grid)

    return nothing
end

@kernel function _barotropic_pressure_correction!(U, grid, Δt, g, η)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U.u[i, j, k] -= g * Δt * ∂xᶠᶜᶠ(i, j, grid.Nz+1, grid, η)
        U.v[i, j, k] -= g * Δt * ∂yᶜᶠᶠ(i, j, grid.Nz+1, grid, η)
    end
end
