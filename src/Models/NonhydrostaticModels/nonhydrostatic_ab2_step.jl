using Oceananigans.TimeSteppers: _ab2_step_field!, implicit_step!
using Oceananigans: instantiated_location
using Oceananigans.Fields: Field
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node
import Oceananigans.TimeSteppers: ab2_step!

@kernel function _ab2_step_immersed_velocity!(u, grid, (ℓx, ℓy, ℓz), Δt, χ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    FT = eltype(u)
    α = convert(FT, 3/2) + χ
    β = convert(FT, 1/2) + χ
    not_euler = χ != convert(FT, -0.5)

    @inbounds begin
        Gu = α * Gⁿ[i, j, k] - β * G⁻[i, j, k] * not_euler
        value = u[i, j, k] + Δt * Gu
        masked = immersed_peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz)
        u[i, j, k] = ifelse(masked, zero(FT), value)
    end
end

@inline function ab2_substep_velocity!(u, grid, Δt, χ, Gⁿ, G⁻, implicit_solver)
    if implicit_solver !== nothing && u isa Field
        launch!(architecture(grid), grid, :xyz, _ab2_step_immersed_velocity!,
                u, grid, instantiated_location(u), Δt, χ, Gⁿ, G⁻; exclude_periphery=true)
    else
        launch!(architecture(grid), grid, :xyz, _ab2_step_field!,
                u, Δt, χ, Gⁿ, G⁻; exclude_periphery=true)
    end

    return nothing
end

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
    @inline substep_velocity!(u, Gⁿ, G⁻) = ab2_substep_velocity!(u, grid, Δt, χ, Gⁿ, G⁻, model.timestepper.implicit_solver)
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
Unrolling the loops over `Val`-wrapped field names keeps each launch type stable.
"""
function step_prognostic_fields!(model, substep_velocity!::SV, substep_tracer!::ST, implicit_Δt) where {SV, ST}
    # `SV` and `ST` force specializing on the closures, which are only passed through: otherwise,
    # when this function is not inlined, they are boxed and the calls below are dispatched dynamically
    implicit_advecting_velocities(model)
    step_velocities!(model, substep_velocity!, implicit_Δt)
    step_tracers!(model, substep_tracer!, implicit_Δt)
    return nothing
end

function step_velocities!(model, substep_velocity!::SV, implicit_Δt) where SV
    foreach_name(model.velocities) do _, val_name
        step_velocity!(model, substep_velocity!, implicit_Δt, val_name)
    end
    return nothing
end

function step_tracers!(model, substep_tracer!::ST, implicit_Δt) where ST
    foreach_name(model.tracers) do val_tracer_index, val_name
        step_tracer!(model, substep_tracer!, implicit_Δt, val_tracer_index, val_name)
    end
    return nothing
end

# `fields(model)` and `advecting_velocities(model)` are rebuilt for every field below:
# passing the tuples down to each call allocates

@inline function step_velocity!(model, substep_velocity!, implicit_Δt, ::Val{name}) where name
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

    return nothing
end

@inline function step_tracer!(model, substep_tracer!, implicit_Δt, ::Val{tracer_index}, ::Val{name}) where {tracer_index, name}
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

    return nothing
end
