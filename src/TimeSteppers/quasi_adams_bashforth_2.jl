using Oceananigans.Fields: FunctionField

mutable struct QuasiAdamsBashforth2TimeStepper{FT, GT, IT} <: AbstractTimeStepper
                  χ :: FT
                 Gⁿ :: GT
                 G⁻ :: GT
    implicit_solver :: IT
end

"""
    QuasiAdamsBashforth2TimeStepper(grid, prognostic_fields, χ = 0.1;
                                    implicit_solver = nothing,
                                    Gⁿ = map(similar, prognostic_fields),
                                    G⁻ = map(similar, prognostic_fields))

Return a 2nd-order quasi Adams-Bashforth (AB2) time stepper (`QuasiAdamsBashforth2TimeStepper`)
on `grid`, with `tracers`, and AB2 parameter `χ`. The tendency fields `Gⁿ` and `G⁻`, usually equal to
the `prognostic_fields` that is passed as positional argument, can be specified via optional `kwargs`.

The 2nd-order quasi Adams-Bashforth timestepper steps forward the state `Uⁿ` by `Δt` via

```julia
Uⁿ⁺¹ = Uⁿ + Δt * [(3/2 + χ) * Gⁿ - (1/2 + χ) * Gⁿ⁻¹]
```

where `Uⁿ` is the state at the ``n``-th timestep, `Gⁿ` is the tendency
at the ``n``-th timestep, and `Gⁿ⁻¹` is the tendency at the previous
timestep (`G⁻`).

!!! note "First timestep"
    For the first timestep, since there are no saved tendencies from the previous timestep,
    the `QuasiAdamsBashforth2TimeStepper` performs an Euler timestep:

    ```julia
    Uⁿ⁺¹ = Uⁿ + Δt * Gⁿ
    ```
"""
function QuasiAdamsBashforth2TimeStepper(grid, prognostic_fields, χ = 0.1;
                                         implicit_solver::IT = nothing,
                                         Gⁿ = map(similar, prognostic_fields),
                                         G⁻ = map(similar, prognostic_fields)) where IT

    FT = eltype(grid)
    GT = typeof(Gⁿ)
    χ  = convert(FT, χ)

    return QuasiAdamsBashforth2TimeStepper{FT, GT, IT}(χ, Gⁿ, G⁻, implicit_solver)
end

reset!(timestepper::QuasiAdamsBashforth2TimeStepper) = nothing

#####
##### Time steppping
#####

"""
    time_step!(model::AbstractModel{<:QuasiAdamsBashforth2TimeStepper}, Δt; euler=false, callbacks=[])

Step forward `model` one time step `Δt` with a 2nd-order Adams-Bashforth method.
Setting `euler=true` will take a forward Euler time step.

The steps of the Quasi-Adams-Bashforth second-order (AB2) algorithm are:

1. If this is the first time step (`model.clock.iteration == 0`), call `update_state!`.
2. Call `ab2_step!(model, Δt, callbacks)` which:
   - Computes tendencies for all prognostic fields
   - Advances fields using AB2: `U += Δt * ((3/2 + χ) * Gⁿ - (1/2 + χ) * G⁻)`
   - Applies model-specific corrections (e.g., pressure correction for incompressibility)
3. Store the current tendencies in `G⁻` for use in the next time step.
4. Update the model state (fill halos, compute diagnostics).
5. Advance the clock and step Lagrangian particles.

The specific implementation of `ab2_step!` varies by model type (e.g., `NonhydrostaticModel`
includes a pressure correction step, while `HydrostaticFreeSurfaceModel` handles the
free surface and barotropic mode).
"""
function time_step!(model::AbstractModel{<:QuasiAdamsBashforth2TimeStepper}, Δt;
                    callbacks=[], euler=false)

    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # Take an euler step if:
    #   * We detect that the time-step size has changed.
    #   * We detect that this is the "first" time-step, which means we
    #     need to take an euler step. Note that model.clock.last_Δt is
    #     initialized as Inf
    #   * The user has passed euler=true to time_step!
    euler = euler || (Δt != model.clock.last_Δt)
    euler && @debug "Taking a forward Euler step."

    if model.clock.iteration == 0
        update_state!(model, callbacks)
    end

    # If euler, then set χ = -0.5
    minus_point_five = convert(eltype(model.grid), -0.5)
    ab2_timestepper = model.timestepper
    χ = ifelse(euler, minus_point_five, ab2_timestepper.χ)
    χ₀ = ab2_timestepper.χ # Save initial value
    ab2_timestepper.χ = χ

    ab2_step!(model, Δt, callbacks)
    cache_previous_tendencies!(model)
    update_state!(model, callbacks)

    tick!(model.clock, Δt)

    step_lagrangian_particles!(model, Δt)

    # Return χ to initial value
    ab2_timestepper.χ = χ₀

    return nothing
end

"""
Time step fields via the 2nd-order quasi Adams-Bashforth method

    `U^{n+1} = U^n + Δt ((3/2 + χ) * G^{n} - (1/2 + χ) G^{n-1})`

"""
@kernel function _ab2_step_field!(u, Δt, χ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    FT = eltype(u)
    Δt = convert(FT, Δt)
    α = 3*one(FT)/2 + χ
    β = 1*one(FT)/2 + χ
    not_euler = χ != convert(FT, -0.5) # use to prevent corruption by leftover NaNs in G⁻

    @inbounds begin
        Gu = α * Gⁿ[i, j, k] - β * G⁻[i, j, k] * not_euler
        u[i, j, k] += Δt * Gu
    end
end

@kernel _ab2_step_field!(::FunctionField, Δt, χ, Gⁿ, G⁻) = nothing

#####
##### These functions need to be implemented by every model independently
#####

"""
    ab2_step!(model::AbstractModel, Δt, callbacks)

Advance the model state by one Adams-Bashforth 2nd-order (AB2) time step of size `Δt`.

This is an abstract interface that must be implemented by each model type
(e.g., `NonhydrostaticModel`, `HydrostaticFreeSurfaceModel`).

The implementation should:
1. Compute tendencies for velocities and tracers
2. Advance prognostic fields using AB2: `U += Δt * ((3/2 + χ) * Gⁿ - (1/2 + χ) * G⁻)`
3. Apply any necessary corrections (e.g., pressure correction for incompressibility)

The AB2 parameter `χ` is stored in `model.timestepper.χ`. When `χ = -0.5`, the scheme
reduces to forward Euler (used for the first time step).
"""
ab2_step!(model::AbstractModel, Δt, callbacks) = error("ab2_step! not implemented for $(typeof(model))")

"""
    cache_previous_tendencies!(model::AbstractModel)

Store the current tendencies `Gⁿ` into `G⁻` for use in the next AB2 time step.

This is an abstract interface that must be implemented by each model type.
Called after advancing the model state but before updating tendencies for the next step.
"""
cache_previous_tendencies!(model::AbstractModel) = error("cache_previous_tendencies! not implemented for $(typeof(model))")
