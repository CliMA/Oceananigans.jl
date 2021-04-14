import Oceananigans.TimeSteppers: calculate_pressure_correction!, pressure_correct_velocities!

calculate_pressure_correction!(::HydrostaticFreeSurfaceModel, Δt) = nothing

#####
##### Barotropic pressure correction for models with a free surface
#####

pressure_correct_velocities!(model::HydrostaticFreeSurfaceModel{T, E, A, <:ExplicitFreeSurface}, Δt) where {T, E, A} = nothing

#####
##### Barotropic pressure correction for models with a free surface
#####

function pressure_correct_velocities!(model::HydrostaticFreeSurfaceModel{T, E, A, <:ImplicitFreeSurface}, Δt) where {T, E, A}

    event = launch!(model.architecture, model.grid, :xyz,
                    _barotropic_pressure_correction,
                    model.velocities,
                    model.grid,
                    Δt,
                    model.free_surface.gravitational_acceleration,
                    model.free_surface.η,
                    dependencies = Event(device(model.architecture)))

    return nothing
end

@kernel function _barotropic_pressure_correction(U, grid, Δt, g, η)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U.u[i, j, k] -= g * Δt * ∂xᶠᶜᵃ(i, j, k, grid, η)
        U.v[i, j, k] -= g * Δt * ∂yᶜᶠᵃ(i, j, k, grid, η)
    end
end
