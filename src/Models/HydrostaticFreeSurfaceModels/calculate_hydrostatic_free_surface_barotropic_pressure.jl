import Oceananigans.TimeSteppers: calculate_pressure_correction!, pressure_correct_velocities!

calculate_pressure_correction!(::HydrostaticFreeSurfaceModel, Δt) = nothing

function pressure_correct_velocities!(model::HydrostaticFreeSurfaceModel, Δt)

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

@kernel function _barotropic_pressure_correction(U★, grid, Δt, g, η)
    i, j, k = @index(Global, NTuple)

    @inbounds U.u[i, j, k] -= g * ∂xᶠᵃᵃ(i, j, k, grid, η)
    @inbounds U.v[i, j, k] -= g * ∂yᵃᶠᵃ(i, j, k, grid, η)
end
