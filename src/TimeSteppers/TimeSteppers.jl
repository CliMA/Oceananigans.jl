module TimeSteppers

export
    QuasiAdamsBashforth2TimeStepper,
    RungeKutta3TimeStepper,
    time_step!,
    Clock,
    tendencies

using CUDA
using KernelAbstractions
using Oceananigans: AbstractModel, initialize!, prognostic_fields
using Oceananigans.Architectures: device
using Oceananigans.Utils: work_layout

"""
    abstract type AbstractTimeStepper

Abstract supertype for time steppers.
"""
abstract type AbstractTimeStepper end

function update_state! end
function compute_tendencies! end

calculate_pressure_correction!(model, Δt) = nothing
pressure_correct_velocities!(model, Δt) = nothing

# Interface for time-stepping Lagrangian particles
abstract type AbstractLagrangianParticles end
step_lagrangian_particles!(model, Δt) = nothing

reset!(timestepper) = nothing
implicit_step!(field, ::Nothing, args...; kwargs...) = nothing

include("clock.jl")
include("store_tendencies.jl")
include("quasi_adams_bashforth_2.jl")
include("runge_kutta_3.jl")
include("split_hydrostatic_runge_kutta_3.jl")

"""
    TimeStepper(name::Symbol, args...; kwargs...)

Returns a timestepper with name `name`, instantiated with `args...` and `kwargs...`.

Example
=======

```julia
julia> stepper = TimeStepper(:QuasiAdamsBashforth2, CPU(), grid, tracernames)
```
"""
TimeStepper(name::Symbol, args...; kwargs...) = TimeStepper(Val(name), args...; kwargs...)

# Fallback
TimeStepper(stepper::AbstractTimeStepper, args...; kwargs...) = stepper

#individual contructors
TimeStepper(::Val{:QuasiAdamsBashforth2}, args...; kwargs...) =
    QuasiAdamsBashforth2TimeStepper(args...; kwargs...)

TimeStepper(::Val{:RungeKutta3}, args...; kwargs...) =
    RungeKutta3TimeStepper(args...; kwargs...)

TimeStepper(::Val{:SplitRungeKutta3}, args...; kwargs...) =
    SplitRungeKutta3TimeStepper(args...; kwargs...)

function first_time_step!(model::AbstractModel, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

function first_time_step!(model::AbstractModel{<:QuasiAdamsBashforth2TimeStepper}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt, euler=true)
    return nothing
end

end # module
