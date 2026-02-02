module TimeSteppers

export
    QuasiAdamsBashforth2TimeStepper,
    RungeKutta3TimeStepper,
    SplitRungeKuttaTimeStepper,
    time_step!,
    Clock

using KernelAbstractions: @kernel, @index
using Oceananigans: AbstractModel, initialize!, prognostic_fields

"""
    abstract type AbstractTimeStepper

Abstract supertype for time steppers.
"""
abstract type AbstractTimeStepper end

function update_state! end
function compute_tendencies! end
function compute_flux_bc_tendencies! end

# Interface for time-stepping Lagrangian particles
abstract type AbstractLagrangianParticles end
step_lagrangian_particles!(model, Δt) = nothing

reset!(timestepper) = nothing
implicit_step!(field, ::Nothing, args...; kwargs...) = nothing

include("clock.jl")
include("quasi_adams_bashforth_2.jl")
include("runge_kutta_3.jl")
include("split_runge_kutta.jl")

"""
    TimeStepper(name::Symbol, args...; kwargs...)

Return a timestepper with name `name`, instantiated with `args...` and `kwargs...`.

Example
=======

```julia
julia> stepper = TimeStepper(:QuasiAdamsBashforth2, grid, tracernames)
```
"""
TimeStepper(name::Symbol, args...; kwargs...) = TimeStepper(Val(name), args...; kwargs...)

# Fallback
TimeStepper(stepper::AbstractTimeStepper, args...; kwargs...) = stepper

#individual constructors
TimeStepper(::Val{:QuasiAdamsBashforth2}, args...; kwargs...) =
    QuasiAdamsBashforth2TimeStepper(args...; kwargs...)

TimeStepper(::Val{:RungeKutta3}, args...; kwargs...) =
    RungeKutta3TimeStepper(args...; kwargs...)

# Convenience constructors for SplitRungeKuttaTimeStepper with 2 to 5 stages
# By calling TimeStepper(:SplitRungeKuttaN, ...)
for stages in 2:5
    @eval TimeStepper(::Val{Symbol(:SplitRungeKutta, $stages)}, args...; kwargs...) =
              SplitRungeKuttaTimeStepper(args...; coefficients=tuple(collect($stages:-1:1)...), kwargs...)
end

TimeStepper(ts::SplitRungeKuttaTimeStepper, grid, prognostic_fields; kw...) =
    SplitRungeKuttaTimeStepper(grid, prognostic_fields; coefficients=ts.β, kw...)

TimeStepper(ts::QuasiAdamsBashforth2TimeStepper, grid, prognostic_fields; kw...) =
    QuasiAdamsBashforth2TimeStepper(grid, prognostic_fields; χ=ts.χ, kw...)

function first_time_step!(model::AbstractModel, Δt)
    initialize!(model)
    # The first update_state! is conditionally gated from within time_step!
    # update_state!(model)
    time_step!(model, Δt)
    return nothing
end

function first_time_step!(model::AbstractModel{<:QuasiAdamsBashforth2TimeStepper}, Δt)
    initialize!(model)
    # The first update_state! is conditionally gated from within time_step!
    # update_state!(model)
    time_step!(model, Δt, euler=true)
    return nothing
end

#####
##### Checkpointing
#####

function prognostic_state(timestepper::AbstractTimeStepper)
    return (Gⁿ = prognostic_state(timestepper.Gⁿ),
            G⁻ = prognostic_state(timestepper.G⁻))
end

function restore_prognostic_state!(restored::AbstractTimeStepper, from)
    restore_prognostic_state!(restored.Gⁿ, from.Gⁿ)
    restore_prognostic_state!(restored.G⁻, from.G⁻)
    return restored
end

restore_prognostic_state!(::AbstractTimeStepper, ::Nothing) = nothing

end # module
