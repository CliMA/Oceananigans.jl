module TimeSteppers

export
    AdamsBashforthTimeStepper,
    time_step!,
    compute_w_from_continuity!

using GPUifyLoops: @launch, @loop, @unroll

import Oceananigans: TimeStepper

using Oceananigans.Architectures: @hascuda
@hascuda using CUDAnative, CuArrays

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

julia> stepper = TimeStepper(:AdamsBashforth, Float64, CPU(), grid, tracernames)
"""
function TimeStepper(name::Symbol, args...)
    fullname = Symbol(name, :TimeStepper)
    return eval(Expr(:call, fullname, args...))
end

# Fallback
TimeStepper(stepper, args...) = stepper

"""Returns the arguments passed to boundary conditions functions."""
boundary_condition_function_arguments(model) =
    (model.clock.time, model.clock.iteration, datatuple(model.velocities),
     datatuple(model.tracers), model.parameters)

#####
##### Time-stepping kernels/function that are independent of the TimeStepper
#####

"""
    calculate_explicit_substep!(tendencies, velocities, tracers, pressures, diffusivities, model)

Calculate the initial and explicit substep of the two-step fractional step method with pressure correction.
"""
function calculate_explicit_substep!(tendencies, velocities, tracers, pressures, diffusivities, model)
    time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)
    calculate_tendencies!(tendencies, velocities, tracers, pressures, diffusivities, model)
    return nothing
end

"""
    time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)

Perform precomputations necessary for an explicit timestep or substep.
"""
function time_step_precomputations!(diffusivities, pressures, velocities, tracers, model)

    fill_halo_regions!(merge(model.velocities, model.tracers), model.architecture,
                       boundary_condition_function_arguments(model)...)

    calculate_diffusivities!(diffusivities, model.architecture, model.grid, model.closure,
                             model.buoyancy, velocities, tracers)

    fill_halo_regions!(model.diffusivities, model.architecture)

    @launch(device(model.architecture), config=launch_config(model.grid, :xy),
            update_hydrostatic_pressure!(pressures.pHY′, model.grid, model.buoyancy, tracers))

    fill_halo_regions!(model.pressures.pHY′, model.architecture)

    return nothing
end

"""
    calculate_tendencies!(diffusivities, pressures, velocities, tracers, model)

Calculate the interior and boundary contributions to tendency terms without the
contribution from non-hydrostatic pressure.
"""
function calculate_tendencies!(tendencies, velocities, tracers, pressures, diffusivities, model)

    calculate_interior_source_terms!(
        tendencies, model.architecture, model.grid, model.coriolis, model.buoyancy,
        model.surface_waves, model.closure, velocities, tracers, pressures.pHY′,
        diffusivities, model.forcing, model.parameters, model.clock.time
    )

    calculate_boundary_source_terms!(
        model.timestepper.Gⁿ, model.architecture, model.velocities,
        model.tracers, boundary_condition_function_arguments(model)...
    )

    return nothing
end

"""
    calculate_pressure_correction!(nonhydrostatic_pressure, Δt, tendencies, velocities, model)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function calculate_pressure_correction!(nonhydrostatic_pressure, Δt, tendencies, velocities, model)
    velocity_tendencies = (u=model.timestepper.Gⁿ.u, v=model.timestepper.Gⁿ.v, w=model.timestepper.Gⁿ.w)

    fill_halo_regions!(velocity_tendencies, model.architecture)

    solve_for_pressure!(nonhydrostatic_pressure, model.pressure_solver,
                        model.architecture, model.grid, velocities, tendencies, Δt)

    fill_halo_regions!(model.pressures.pNHS, model.architecture)

    return nothing
end

calculate_pressure_correction!(::Nothing, args...) = nothing

"""
    complete_pressure_correction_step!(velocities, Δt, tracers, pressures, tendencies, model)

After calculating the pressure correction, complete the pressure correction step by updating
the velocity and tracer fields.
"""
function complete_pressure_correction_step!(velocities, Δt, tracers, pressures, tendencies, model)

    update_solution!(velocities, tracers, model.architecture,
                     model.grid, Δt, tendencies, pressures.pNHS)

    fill_halo_regions!(model.velocities, model.architecture,
                       boundary_condition_function_arguments(model)...)

    compute_w_from_continuity!(model)

    return nothing
end

include("kernels.jl")
include("adams_bashforth.jl")

end # module
