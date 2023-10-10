module TimeSteppers

export
    QuasiAdamsBashforth2TimeStepper,
    RungeKutta3TimeStepper,
    time_step!,
    Clock,
    tendencies

using CUDA
using KernelAbstractions
using Oceananigans: AbstractModel, prognostic_fields
using Oceananigans.Architectures: device
using Oceananigans.Fields: TendencyFields
using Oceananigans.Utils: work_layout

"""
    abstract type AbstractTimeStepper

Abstract supertype for time steppers.
"""
abstract type AbstractTimeStepper end

"""
    TimeStepper(name::Symbol, args...; kwargs...)

Returns a timestepper with name `name`, instantiated with `args...` and `kwargs...`.

Example
=======

```julia
julia> stepper = TimeStepper(:QuasiAdamsBashforth2, CPU(), grid, tracernames)
```
"""
function TimeStepper(name::Symbol, args...; kwargs...)
    fullname = Symbol(name, :TimeStepper)
    return @eval $fullname($args...; $kwargs...)
end

# Fallback
TimeStepper(stepper::AbstractTimeStepper, args...; kwargs...) = stepper

function update_state! end
function compute_tendencies! end

calculate_pressure_correction!(model, Δt) = nothing
pressure_correct_velocities!(model, Δt) = nothing

# Interface for time-stepping Lagrangian particles
abstract type AbstractLagrangianParticles end
step_lagrangian_particles!(model, Δt) = nothing

reset!(timestepper) = nothing

include("clock.jl")
include("store_tendencies.jl")
include("quasi_adams_bashforth_2.jl")
include("runge_kutta_3.jl")

end # module
