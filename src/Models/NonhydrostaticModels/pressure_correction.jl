import Oceananigans.TimeSteppers: calculate_pressure_correction!, pressure_correct_velocities!

using Oceananigans.Models.HydrostaticFreeSurfaceModels: step_free_surface!

"""
    calculate_pressure_correction!(model::NonhydrostaticModel, Δt)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function calculate_pressure_correction!(model::NonhydrostaticModel, Δt)

    if !isnothing(model.free_surface)
        step_free_surface!(model.free_surface, model, model.timestepper, Δt)
        # "First" barotropic pressure correction
        pressure_correct_velocities!(model, model.free_surface, Δt)
    end

    # Mask immersed velocities
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities, model.clock, fields(model))

    solve_for_pressure!(model.pressures.pNHS, model.pressure_solver, Δt, model.velocities)
    fill_halo_regions!(model.pressures.pNHS)

    return nothing
end

#####
##### Fractional and time stepping
#####

"""
Update the predictor velocities u, v, and w with the non-hydrostatic pressure via

    `u^{n+1} = u^n - δₓp_{NH} / Δx * Δt`
"""
@kernel function _pressure_correct_velocities!(U, grid, Δt, pNHS)
    i, j, k = @index(Global, NTuple)

    @inbounds U.u[i, j, k] -= ∂xᶠᶜᶜ(i, j, k, grid, pNHS) * Δt
    @inbounds U.v[i, j, k] -= ∂yᶜᶠᶜ(i, j, k, grid, pNHS) * Δt
    @inbounds U.w[i, j, k] -= ∂zᶜᶜᶠ(i, j, k, grid, pNHS) * Δt
end

"Update the solution variables (velocities and tracers)."
function pressure_correct_velocities!(model::NonhydrostaticModel, Δt)

    launch!(model.architecture, model.grid, :xyz,
            _pressure_correct_velocities!,
            model.velocities,
            model.grid,
            Δt,
            model.pressures.pNHS)

    return nothing
end
