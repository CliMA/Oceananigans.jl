using Oceananigans.TimeSteppers: _rk3_substep_field!, _rk3_substep_fields!, stage_Δt
import Oceananigans.TimeSteppers: rk3_substep!

"""
    rk3_substep!(model::NonhydrostaticModel, Δt, γ, ζ, callbacks)

Perform a single RK3 substep for `NonhydrostaticModel` with pressure correction.
Dispatches to `pressure_correction_rk3_substep!` which advances velocities and tracers
using the RK3 coefficients, then applies a pressure correction to enforce incompressibility.
"""
rk3_substep!(model::NonhydrostaticModel, Δt, γ, ζ, callbacks) =
    pressure_correction_rk3_substep!(model, Δt, γ, ζ, callbacks)

"""
    pressure_correction_rk3_substep!(model, Δt, γⁿ, ζⁿ, callbacks)

Implement a single RK3 substep with pressure correction for `NonhydrostaticModel`.

The substep advances the state as

    U += Δt * (γⁿ * Gⁿ + ζⁿ * G⁻)

where:
- `γⁿ` is the coefficient for the current tendency
- `ζⁿ` is the coefficient for the previous tendency (or `nothing` for the first substep)
- The effective substep size is `Δτ = Δt * (γⁿ + ζⁿ)`

After advancing velocities, a pressure Poisson equation is solved and velocities
are corrected to satisfy the incompressibility constraint.
"""
function pressure_correction_rk3_substep!(model, Δt, γⁿ, ζⁿ, callbacks)
    Δτ   = stage_Δt(Δt, γⁿ, ζⁿ)
    grid = model.grid

    compute_flux_bc_tendencies!(model)
    model_fields = prognostic_fields(model)

    launch!(architecture(grid), grid, :xyz, 
            _rk3_substep_fields!, 
            values(model.velocities), 
            Δt, γⁿ, ζⁿ, 
            values(model.timestepper.Gⁿ[(:u, :v, :w)]), 
            values(model.timestepper.G⁻[(:u, :v, :w)]); 
            exclude_periphery = true)

    tracer_keys = keys(model.tracers)

    launch!(architecture(grid), grid, :xyz, 
            _rk3_substep_fields!, 
            values(model.tracers), 
            Δt, γⁿ, ζⁿ, 
            values(model.timestepper.Gⁿ[tracer_keys]), 
            values(model.timestepper.G⁻[tracer_keys]); 
            exclude_periphery = false)

    for (i, name) in enumerate(keys(model_fields))
        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       Val(i-3), # We assume that the first 3 fields are velocity / momentum variables
                       model.clock,
                       fields(model),
                       Δτ)
    end

    compute_pressure_correction!(model, Δτ)
    make_pressure_correction!(model, Δτ)

    return nothing
end
