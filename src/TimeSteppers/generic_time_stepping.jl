#####
##### Time-stepping functionality that is independent of the TimeStepper
#####

"""
    calculate_explicit_substep!(tendencies, velocities, tracers, pressures, diffusivities, model)

Calculate the initial and explicit substep of the two-step fractional step method with pressure correction.
"""
function calculate_explicit_substep!(tendencies, velocities, tracers, pressures, diffusivities, model)
    time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)
    calculate_tendencies!(tendencies, velocities, tracers, pressures, diffusivities, model)
    return nothing
end

"""
    time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)

Perform precomputations necessary for an explicit timestep or substep.
"""
function time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)

    fill_halo_regions!(merge(model.velocities, model.tracers), model.architecture,
                       boundary_condition_function_arguments(model)...)

    calculate_diffusivities!(diffusivities, model.architecture, model.grid, model.closure,
                             model.buoyancy, velocities, tracers)

    fill_halo_regions!(model.diffusivities, model.architecture)

    @launch(device(model.architecture), config=launch_config(model.grid, :xy),
            update_hydrostatic_pressure!(pressures.pHY′, model.grid, model.buoyancy, tracers))

    fill_halo_regions!(model.pressures.pHY′, model.architecture)

    return nothing
end

"""
    calculate_tendencies!(diffusivities, pressures, velocities, tracers, model)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(tendencies, velocities, tracers, pressures, diffusivities, model)

    calculate_interior_source_terms!(
        tendencies, model.architecture, model.grid, model.coriolis, model.buoyancy,
        model.surface_waves, model.closure, velocities, tracers, pressures.pHY′,
        diffusivities, model.forcing, model.parameters, model.clock.time
    )

    calculate_boundary_source_terms!(
        model.timestepper.Gⁿ, model.architecture, model.velocities,
        model.tracers, boundary_condition_function_arguments(model)...
    )

    return nothing
end

"""
    calculate_pressure_correction!(nonhydrostatic_pressure, Δt, tendencies, velocities, model)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function calculate_pressure_correction!(nonhydrostatic_pressure, Δt, predictor_velocities, model)
    fill_halo_regions!(model.timestepper.predictor_velocities, model.architecture,
                       boundary_condition_function_arguments(model)...)

    solve_for_pressure!(nonhydrostatic_pressure, model.pressure_solver,
                        model.architecture, model.grid, Δt, predictor_velocities)

    fill_halo_regions!(model.pressures.pNHS, model.architecture)

    return nothing
end

#####
##### Fractional and time stepping
#####

"""
Update the horizontal velocities u and v via

    `u^{n+1} = u^n + (Gu^{n+½} - δₓp_{NH} / Δx) Δt`

Note that the vertical velocity is not explicitly time stepped.
"""
function _fractional_step_velocities!(U, grid, Δt, pNHS)
    @loop_xyz i j k grid begin
        @inbounds U.u[i, j, k] -= ∂xᶠᵃᵃ(i, j, k, grid, pNHS) * Δt
        @inbounds U.v[i, j, k] -= ∂yᵃᶠᵃ(i, j, k, grid, pNHS) * Δt
    end
    return nothing
end

"Update the solution variables (velocities and tracers)."
function fractional_step_velocities!(U, C, arch, grid, Δt, pNHS)
    @launch device(arch) config=launch_config(grid, :xyz) _fractional_step_velocities!(U, grid, Δt, pNHS)
    return nothing
end

