using Oceananigans.TimeSteppers: _rk3_substep_field!, stage_Δt, compute_flux_bc_tendencies!
import Oceananigans.TimeSteppers: rk3_substep!

"""
$(TYPEDSIGNATURES)

Perform a single RK3 substep for `NonhydrostaticModel` with pressure correction.
Dispatches to `pressure_correction_rk3_substep!` which advances velocities and tracers
using the RK3 coefficients, then applies a pressure correction to enforce incompressibility.
"""
rk3_substep!(model::NonhydrostaticModel, Δt, γ, ζ, callbacks) =
    pressure_correction_rk3_substep!(model, Δt, γ, ζ, callbacks)

"""
$(TYPEDSIGNATURES)

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
    grid = model.grid
    Δτ = stage_Δt(Δt, γⁿ, ζⁿ)

    compute_flux_bc_tendencies!(model)

    # Prognostic variables stepping
    @inline substep_velocity!(u, Gⁿ, G⁻) = launch!(architecture(grid), grid, :xyz, _rk3_substep_field!, u, Δt, γⁿ, ζⁿ, Gⁿ, G⁻; exclude_periphery=true)
    @inline substep_tracer!(c, Gⁿ, G⁻)   = launch!(architecture(grid), grid, :xyz, _rk3_substep_field!, c, Δt, γⁿ, ζⁿ, Gⁿ, G⁻)

    step_prognostic_fields!(model, substep_velocity!, substep_tracer!, Δτ)

    compute_pressure_correction!(model, Δτ)
    make_pressure_correction!(model, Δτ)

    return nothing
end
