module TimeSteppers

export
    QuasiAdamsBashforth2TimeStepper,
    RungeKutta3TimeStepper,
    time_step!,
    tendencies

using CUDA
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll

import Oceananigans: TimeStepper

using Oceananigans.Architectures: @hascuda, device
using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Operators
using Oceananigans.Coriolis
using Oceananigans.Buoyancy
using Oceananigans.SurfaceWaves
using Oceananigans.BoundaryConditions
using Oceananigans.Solvers
using Oceananigans.Models
using Oceananigans.Utils

using Oceananigans.TurbulenceClosures:
    calculate_diffusivities!, ∂ⱼ_2ν_Σ₁ⱼ, ∂ⱼ_2ν_Σ₂ⱼ, ∂ⱼ_2ν_Σ₃ⱼ, ∇_κ_∇c

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

include("precomputations.jl")
include("pressure_correction.jl")
include("velocity_and_tracer_tendencies.jl")
include("calculate_tendencies.jl")
include("store_tendencies.jl")
include("update_hydrostatic_pressure.jl")
include("quasi_adams_bashforth_2.jl")
include("runge_kutta_3.jl")

end # module
