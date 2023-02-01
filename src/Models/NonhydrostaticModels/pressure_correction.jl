using Oceananigans.ImmersedBoundaries: mask_immersed_velocities!, mask_immersed_field!

import Oceananigans.TimeSteppers: calculate_pressure_correction!, pressure_correct_velocities!

"""
    calculate_pressure_correction!(model::NonhydrostaticModel, Δt)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function calculate_pressure_correction!(model::NonhydrostaticModel, Δt)

    # Mask immersed velocities
    velocity_masking_events = mask_immersed_velocities!(model.velocities, model.architecture, model.grid)
    wait(device(model.architecture), MultiEvent(velocity_masking_events))

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
    
    @inbounds U.u[i, j, k] -= δxᶠᵃᵃ(i, j, k, grid, pNHS) / Δxᶠᶜᶜ(i, j, k, grid) * Δt
    @inbounds U.v[i, j, k] -= δyᵃᶠᵃ(i, j, k, grid, pNHS) / Δyᶜᶠᶜ(i, j, k, grid) * Δt
    @inbounds U.w[i, j, k] -= δzᵃᵃᶠ(i, j, k, grid, pNHS) / Δzᶜᶜᶠ(i, j, k, grid) * Δt
end

"Update the solution variables (velocities and tracers)."
function pressure_correct_velocities!(model::NonhydrostaticModel, Δt)

    event = launch!(model.architecture, model.grid, :xyz,
                    _pressure_correct_velocities!,
                    model.velocities,
                    model.grid,
                    Δt,
                    model.pressures.pNHS,
                    dependencies = device_event(model.architecture)) 

    wait(device(model.architecture), event)

    return nothing
end
