using Oceananigans.TimeSteppers: rk3_substep_field!, stage_Δt	
import Oceananigans.TimeSteppers: rk3_substep!	

"""
    rk3_substep!(model::NonhydrostaticModel, Δt, γⁿ, ζⁿ, callbacks)

Perform a single RK3 substep for `NonhydrostaticModel` with pressure correction.
Dispatches to `pressure_correction_rk3_substep!` which advances velocities and tracers
using the RK3 coefficients, then applies a pressure correction to enforce incompressibility.
"""
rk3_substep!(model::NonhydrostaticModel, Δt, γ, ζ, callbacks) = 
    pressure_correction_rk3_substep!(model, Δt, γ, ζ, callbacks)

"""
    pressure_correction_rk3_substep!(model, Δt, γⁿ, ζⁿ, callbacks)

Implement a single RK3 substep with pressure correction for `NonhydrostaticModel`.

The substep advances the state as `U += Δt * (γⁿ * Gⁿ + ζⁿ * G⁻)` where:
- `γⁿ` is the coefficient for the current tendency
- `ζⁿ` is the coefficient for the previous tendency (or `nothing` for the first substep)
- The effective substep size is `Δτ = Δt * (γⁿ + ζⁿ)`

After advancing velocities, a pressure Poisson equation is solved and velocities
are corrected to satisfy the incompressibility constraint.
"""
function pressure_correction_rk3_substep!(model, Δt, γⁿ, ζⁿ, callbacks)
    Δτ   = stage_Δt(Δt, γⁿ, ζⁿ) 	
    grid = model.grid	

    # Velocity steps	
    for (i, field) in enumerate(model.velocities)	
        kernel_args = (field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[i], model.timestepper.G⁻[i])	
        launch!(architecture(grid), grid, :xyz, rk3_substep_field!, kernel_args...; exclude_periphery=true)	

        implicit_step!(field,	
                       model.timestepper.implicit_solver,	
                       model.closure,	
                       model.closure_fields,	
                       nothing,	
                       model.clock,	
                       fields(model),	
                       Δτ)	
    end	

    # Tracer steps	
    for (i, name) in enumerate(propertynames(model.tracers))	
        field = model.tracers[name]	
        kernel_args = (field, Δt, γⁿ, ζⁿ, model.timestepper.Gⁿ[name], model.timestepper.G⁻[name])	
        launch!(architecture(grid), grid, :xyz, rk3_substep_field!, kernel_args...)	

        implicit_step!(field,	
                       model.timestepper.implicit_solver,	
                       model.closure,	
                       model.closure_fields,	
                       Val(i),	
                       model.clock,	
                       fields(model),	
                       Δτ)	
    end	

    compute_pressure_correction!(model, Δτ)	
    make_pressure_correction!(model, Δτ)	

    return nothing	
end	
