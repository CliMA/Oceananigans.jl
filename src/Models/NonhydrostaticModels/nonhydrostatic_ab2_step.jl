using Oceananigans.TimeSteppers: _ab2_step_field!, implicit_step!
import Oceananigans.TimeSteppers: ab2_step!

"""
$(TYPEDSIGNATURES)

Advance `NonhydrostaticModel` by one Adams-Bashforth 2nd-order time step with pressure correction.
Dispatches to `pressure_correction_ab2_step!` which implements a predictor-corrector scheme
"""
ab2_step!(model::NonhydrostaticModel, args...) =
    pressure_correction_ab2_step!(model, args...)

"""
$(TYPEDSIGNATURES)

Implement the AB2 time step with pressure correction for `NonhydrostaticModel`.

This predictor-corrector scheme:
1. Computes tendencies `Gⁿ` for all prognostic fields
2. Advances velocities: `u* = uⁿ + Δt * AB2(Gᵤ)` (predictor step)
3. Advances tracers: `cⁿ⁺¹ = cⁿ + Δt * AB2(Gᶜ)`
4. Applies implicit vertical diffusion (if configured)
5. Solves `∇²p = ∇·u* / Δt` for pressure correction
6. Corrects velocities: `uⁿ⁺¹ = u* - Δt * ∇p` to satisfy `∇·uⁿ⁺¹ = 0`
"""
function pressure_correction_ab2_step!(model, Δt, callbacks)
    grid = model.grid

    # Compute flux bc tendencies
    compute_flux_bc_tendencies!(model)

    # Prognostic variables stepping
    χ = model.timestepper.χ
    @inline substep_velocity!(u, Gⁿ, G⁻) = launch!(architecture(grid), grid, :xyz, _ab2_step_field!, u, Δt, χ, Gⁿ, G⁻; exclude_periphery=true)
    @inline substep_tracer!(c, Gⁿ, G⁻)   = launch!(architecture(grid), grid, :xyz, _ab2_step_field!, c, Δt, χ, Gⁿ, G⁻)

    step_prognostic_fields!(model, substep_velocity!, substep_tracer!, Δt)

    compute_pressure_correction!(model, Δt)
    make_pressure_correction!(model, Δt)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Advance the velocities of `model` with `substep_velocity!(u, Gⁿ, G⁻)` and its tracers with
`substep_tracer!(c, Gⁿ, G⁻)`, then apply implicit vertical diffusion over `implicit_Δt`.
The recursions over `Val`-wrapped field names keep each launch type stable.
"""
function step_prognostic_fields!(model, substep_velocity!, substep_tracer!, implicit_Δt)
    implicit_advecting_velocities(model)
    step_velocities!(model, substep_velocity!, implicit_Δt, Val(keys(model.velocities)))
    step_tracers!(model, substep_tracer!, implicit_Δt, Val(1), Val(keys(model.tracers)))
    return nothing
end

# `fields(model)` and `advecting_velocities(model)` are rebuilt at every level of the recursions
# below: passing the tuples down the recursion allocates

@inline step_velocities!(model, substep_velocity!, implicit_Δt, ::Val{()}) = nothing

@inline function step_velocities!(model, substep_velocity!, implicit_Δt, ::Val{names}) where names
    name = first(names)
    u  = model.velocities[name]
    Gⁿ = model.timestepper.Gⁿ[name]
    G⁻ = model.timestepper.G⁻[name]
    substep_velocity!(u, Gⁿ, G⁻)

    implicit_step!(u,
                   model.timestepper.implicit_solver,
                   model.closure,
                   model.closure_fields,
                   nothing,
                   model.clock,
                   fields(model),
                   implicit_Δt,
                   model.advection.momentum,
                   advecting_velocities(model))

    step_velocities!(model, substep_velocity!, implicit_Δt, Val(Base.tail(names)))
    return nothing
end

@inline step_tracers!(model, substep_tracer!, implicit_Δt, ::Val, ::Val{()}) = nothing

@inline function step_tracers!(model, substep_tracer!, implicit_Δt, ::Val{tracer_index}, ::Val{names}) where {tracer_index, names}
    name = first(names)
    c  = model.tracers[name]
    Gⁿ = model.timestepper.Gⁿ[name]
    G⁻ = model.timestepper.G⁻[name]
    substep_tracer!(c, Gⁿ, G⁻)

    implicit_step!(c,
                   model.timestepper.implicit_solver,
                   model.closure,
                   model.closure_fields,
                   Val(tracer_index),
                   model.clock,
                   fields(model),
                   implicit_Δt,
                   model.advection[name],
                   advecting_velocities(model))

    step_tracers!(model, substep_tracer!, implicit_Δt, Val(tracer_index + 1), Val(Base.tail(names)))
    return nothing
end
