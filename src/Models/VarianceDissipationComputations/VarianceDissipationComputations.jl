module VarianceDissipationComputations

export VarianceDissipation, flatten_dissipation_fields

using KernelAbstractions: @kernel, @index
using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans: Oceananigans, UpdateStateCallsite, TracerTendencyCallsite
using Oceananigans.Advection
using Oceananigans.Advection: _advective_tracer_flux_x,
                              _advective_tracer_flux_y,
                              _advective_tracer_flux_z
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.Fields: VelocityFields
using Oceananigans.Grids: architecture, AbstractGrid, peripheral_node
using Oceananigans.Operators
using Oceananigans.TimeSteppers: MultiStageTimeStepper,
                                 QuasiAdamsBashforth2TimeStepper,
                                 RungeKutta3TimeStepper,
                                 SplitRungeKuttaTimeStepper,
                                 SSPRungeKuttaTimeStepper,
                                 ssp_quadrature_weights,
                                 time_discretization
using Oceananigans.TurbulenceClosures: _diffusive_flux_x,
                                       _diffusive_flux_y,
                                       _diffusive_flux_z
using OffsetArrays: OffsetArrays
using Oceananigans.Utils
using Oceananigans.Utils: IterationInterval, ConsecutiveIterations

struct VarianceDissipation{P, K, A, D, S}
    advective_production :: P
    diffusive_production :: K
    advective_fluxes :: A
    diffusive_fluxes :: D
    previous_state :: S
    tracer_name :: Symbol
end

Oceananigans.prognostic_state(::VarianceDissipation) = nothing

function c_grid_vector(grid)
    x = XFaceField(grid)
    y = YFaceField(grid)
    z = ZFaceField(grid)
    return (; x, y, z)
end

"""
    VarianceDissipation(tracer_name, grid)

Construct a `VarianceDissipation` object for a tracer called `tracer_name` that lives on `grid`.
This function computes the variance dissipation diagnostics for the specified tracer in the model.
These include the numerical dissipation implicit to the advection scheme and the explicit
dissipation associated to closures.

This diagnostic is especially useful for models that use a dissipative advection scheme
like [`WENO`](@ref) or [`UpwindBiased`](@ref).

Add it to a simulation with `add_callback!(simulation, dissipation, schedule)`. This registers two callbacks:
one caches the fluxes and the transport with which the stages compute the tracer tendency, the other assembles
the dissipation once the tracer is updated at the end of the step.

Arguments
=========

- `tracer_name`: The name of the tracer for which variance dissipation is computed. This should
                 be a `Symbol`. When calling `ϵ::VarianceDissipation` on the model, this name is
                 used to identify the tracer in the model's state.
- `grid`: The grid on which the tracer is defined.

!!! compat "Time stepper compatibility"
    At the moment, the variance dissipation diagnostic is supported only for a [`QuasiAdamsBashforth2TimeStepper`](@ref),
    a [`SplitRungeKuttaTimeStepper`](@ref) and an [`SSPRungeKuttaTimeStepper`](@ref).
"""
function VarianceDissipation(tracer_name, grid)

    P    = c_grid_vector(grid)
    K    = c_grid_vector(grid)
    Vⁿ   = c_grid_vector(grid)
    Vⁿ⁻¹ = c_grid_vector(grid)
    Fⁿ   = c_grid_vector(grid)
    Fⁿ⁻¹ = c_grid_vector(grid)
    κᶻ   = ZFaceField(grid)
    cⁿ⁻¹ = CenterField(grid)
    Uⁿ⁻¹ = VelocityFields(grid)
    Uⁿ   = VelocityFields(grid)

    previous_state   = (; cⁿ⁻¹, Uⁿ⁻¹, Uⁿ)
    advective_fluxes = (; Fⁿ, Fⁿ⁻¹)
    diffusive_fluxes = (; Vⁿ, Vⁿ⁻¹, κᶻ)

    return VarianceDissipation(P, K, advective_fluxes, diffusive_fluxes, previous_state, tracer_name)
end

final_stage(::QuasiAdamsBashforth2TimeStepper) = 1
final_stage(timestepper::MultiStageTimeStepper) = timestepper.Nstages

function (ϵ::VarianceDissipation)(model, ::TracerTendencyCallsite)
    if model.clock.stage == 1
        parent(ϵ.previous_state.cⁿ⁻¹) .= parent(model.tracers[ϵ.tracer_name])
    end

    cache_fluxes!(ϵ, model, ϵ.tracer_name)

    return nothing
end

function (ϵ::VarianceDissipation)(model, ::UpdateStateCallsite)
    if model.clock.stage == final_stage(model.timestepper)
        compute_dissipation!(ϵ, model, ϵ.tracer_name)
    end

    return nothing
end

@inline getadvection(advection, tracer_name) = advection
@inline getadvection(advection::NamedTuple, tracer_name) = @inbounds advection[tracer_name]

const f = Face()
const c = Center()

include("update_fluxes.jl")
include("advective_dissipation.jl")
include("diffusive_dissipation.jl")
include("compute_dissipation.jl")
include("flatten_dissipation_fields.jl")

using Oceananigans.Simulations: GenericName, generic_callback_name

import Oceananigans.Simulations: Callback, add_callback!, validate_schedule

function Callback(::VarianceDissipation, schedule=IterationInterval(1); kwargs...)
    throw(ArgumentError("add a VarianceDissipation to a simulation with `add_callback!(simulation, dissipation, schedule)`."))
end

# Each callback receives its callsite as parameter, which selects the method of `VarianceDissipation` it calls
function add_callback!(simulation, dissipation::VarianceDissipation, schedule=IterationInterval(1); name=GenericName())
    model = simulation.model

    if model.timestepper isa RungeKutta3TimeStepper
        throw(ArgumentError("VarianceDissipation using a RungeKutta3TimeStepper is not supported."))
    end

    if !hasproperty(model, :velocities)
        throw(ArgumentError("Model must have a velocity field."))
    end

    if !hasproperty(model.tracers, dissipation.tracer_name)
        throw(ArgumentError("Model must have a tracer called $(dissipation.tracer_name)."))
    end

    name = generic_callback_name(name, keys(simulation.callbacks))
    schedule = validate_schedule(dissipation, schedule)

    simulation.callbacks[Symbol(name, :_fluxes)] = Callback(dissipation, schedule, TracerTendencyCallsite(), TracerTendencyCallsite())
    simulation.callbacks[name] = Callback(dissipation, schedule, UpdateStateCallsite(), UpdateStateCallsite())

    return nothing
end

validate_schedule(::VarianceDissipation, schedule) = throw(ArgumentError("the provided schedule $schedule is not supported for VarianceDissipation computations. \n" *
                                                                         "Use an `IterationInterval` schedule instead."))

function validate_schedule(::VarianceDissipation, schedule::IterationInterval)
    if !(schedule == IterationInterval(1))
        @warn "VarianceDissipation callback must be called every Iteration or on `ConsecutiveIterations`. \n" *
              "Changing `schedule` to `ConsecutiveIterations(schedule)`."
        schedule = ConsecutiveIterations(schedule)
    end
    return schedule
end

function validate_schedule(::VarianceDissipation, schedule::ConsecutiveIterations)
    if !(schedule.parent isa IterationInterval)
       throw(ArgumentError("the provided schedule $schedule is not supported for VarianceDissipation computations. \n" *
                                                                         "Use an `IterationInterval` schedule instead."))
    end
    return schedule
end

end # module
