module TimeSteppers

export
    QuasiAdamsBashforth2TimeStepper,
    RungeKutta3TimeStepper,
    time_step!,
    tendencies

using CUDA
using KernelAbstractions
using Oceananigans: AbstractModel, tick!
using Oceananigans.Architectures: @hascuda, device
using Oceananigans.Fields: TendencyFields
using Oceananigans.Utils: work_layout

import Oceananigans: TimeStepper

"""
    AbstractTimeStepper

Abstract supertype for time steppers.
"""
abstract type AbstractTimeStepper end

"""
    TimeStepper(name, args...)

Returns a timestepper with name `name`, instantiated with `args...`.

Example
=======

julia> stepper = TimeStepper(:QuasiAdamsBashforth2, CPU(), grid, tracernames)
"""
function TimeStepper(name::Symbol, args...)
    fullname = Symbol(name, :TimeStepper)
    return eval(Expr(:call, fullname, args...))
end

# Fallbacks
TimeStepper(stepper::AbstractTimeStepper, args...) = stepper

function update_state! end
function calculate_tendencies! end
function calculate_pressure_correction! end
function pressure_correct_velocities! end

#calculate_pressure_correction!(model, Δt) = nothing
#pressure_correct_velocities!(model, Δt) = nothing

include("store_tendencies.jl")
include("quasi_adams_bashforth_2.jl")
include("runge_kutta_3.jl")

end # module
