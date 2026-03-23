module TimeSteppers

using ..Architectures: ReactantState
using ..Grids: ShardedGrid, ReactantGrid

using Reactant
using Oceananigans

using Oceananigans: AbstractModel, Distributed
using Oceananigans.Grids: architecture
using Oceananigans.TimeSteppers:
    update_state!,
    tick!,
    tick_stage!,
    step_lagrangian_particles!,
    QuasiAdamsBashforth2TimeStepper,
    cache_previous_tendencies!

import Oceananigans.TimeSteppers: Clock, first_time_step!, time_step!,
                                  ab2_step!, maybe_initialize_state!
import Oceananigans: initialize!

const ReactantModel{TS} = Union{
    AbstractModel{TS, <:ReactantState},
    AbstractModel{TS, <:Distributed{<:ReactantState}}
}

function Clock(grid::ReactantGrid)
    FT = Oceananigans.defaults.FloatType
    arch = architecture(grid)

    sharding = if arch isa Distributed
        Sharding.Replicated(arch.connectivity)
    else
        Sharding.NoSharding()
    end

    t = ConcreteRNumber(zero(FT), sharding=sharding)
    iter = ConcreteRNumber(0, sharding=sharding)
    stage = 1
    last_Δt = ConcreteRNumber(convert(FT, Inf), sharding=sharding)
    last_stage_Δt = ConcreteRNumber(convert(FT, Inf), sharding=sharding)

    return Clock(; time=t, iteration=iter, stage, last_Δt, last_stage_Δt)
end

innertype(::ConcreteRNumber{T}) where T = T

const ConcreteReactantClock = Clock{<:ConcreteRNumber}
const TracedReactantClock = Oceananigans.TimeSteppers.Clock{<:Reactant.TracedRNumber}

function Base.setproperty!(clock::ConcreteReactantClock, prop::Symbol, value)
    clock_val = getproperty(clock, prop)

    if prop in (:last_Δt, :last_stage_Δt, :time, :iteration)
        converted_val = convert(innertype(clock_val), value)
        if Reactant.Sharding.is_sharded(clock_val)
            sharding = clock_val.sharding
            sharded_val = ConcreteRNumber(converted_val; sharding)
            return setfield!(clock, prop, sharded_val)
        end
    end

    return setfield!(clock, prop, convert(typeof(clock_val), value))
end

# Reactant handles initialization via first_time_step!, so this is a no-op.
maybe_initialize_state!(::ReactantModel, callbacks) = nothing

#####
##### QuasiAdamsBashforth2TimeStepper for Reactant
#####
# AB2 needs a Reactant override because the src/ code does:
#   euler = euler | (Δt != model.clock.last_Δt)
# which makes `euler` a TracedRNumber{Bool}, then ifelse(traced_bool, ...)
# returns TracedRNumber{Float64}, and `ab2_timestepper.χ = TracedRNumber{Float64}`
# fails because the field type is Float64.

function time_step!(model::ReactantModel{<:QuasiAdamsBashforth2TimeStepper{FT}}, Δt;
                    callbacks=[], euler=false) where FT

    # If euler, then set χ = -0.5
    minus_point_five = convert(FT, -0.5)
    ab2_timestepper = model.timestepper
    χ = ifelse(euler, minus_point_five, ab2_timestepper.χ)
    χ₀ = ab2_timestepper.χ # Save initial value
    ab2_timestepper.χ = χ

    ab2_step!(model, Δt, callbacks)
    cache_previous_tendencies!(model)

    tick!(model.clock, Δt)

    update_state!(model, callbacks)
    step_lagrangian_particles!(model, Δt)

    # Return χ to initial value
    ab2_timestepper.χ = χ₀

    return nothing
end

#####
##### first_time_step! for Reactant
#####

function first_time_step!(model::ReactantModel, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

function first_time_step!(model::ReactantModel{<:Oceananigans.TimeSteppers.QuasiAdamsBashforth2TimeStepper}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt, euler=true)
    return nothing
end

function Oceananigans.TimeSteppers.tick_time!(clock::Oceananigans.TimeSteppers.Clock{<:Reactant.TracedRNumber}, Δt)
    t_next = Oceananigans.TimeSteppers.next_time(clock, Δt)
    clock.time.mlir_data = t_next.mlir_data
    return t_next
end

# Promote a value to TracedRNumber via addition with zero(clock.time).
# This is needed because .mlir_data only exists on TracedRNumber.
promote_to_traced(Δt, clock) = Δt + zero(clock.time)

function Oceananigans.TimeSteppers.tick!(clock::TracedReactantClock, Δt)
    Oceananigans.TimeSteppers.tick_time!(clock, Δt)

    clock.iteration.mlir_data = (clock.iteration + 1).mlir_data
    clock.stage = 1
    
    Δt = promote_to_traced(Δt, clock)
    clock.last_Δt.mlir_data = Δt.mlir_data

    # Add zero to avoid aliasing last_stage_Δt with last_Δt
    clock.last_stage_Δt.mlir_data = (Δt + 0).mlir_data
    
    return nothing
end

function Oceananigans.TimeSteppers.tick_stage!(clock::TracedReactantClock, stage_Δt)
    Oceananigans.TimeSteppers.tick_time!(clock, stage_Δt)
    stage_Δt = promote_to_traced(stage_Δt, clock)
    clock.stage += 1
    clock.last_stage_Δt.mlir_data = stage_Δt.mlir_data
    return nothing
end

function Oceananigans.TimeSteppers.tick_stage!(clock::TracedReactantClock, stage_Δt, step_Δt)
    Oceananigans.TimeSteppers.tick_time!(clock, stage_Δt)
    stage_Δt = promote_to_traced(stage_Δt, clock)
    step_Δt = promote_to_traced(step_Δt, clock)
    clock.iteration.mlir_data = (clock.iteration + 1).mlir_data
    clock.stage = 1
    clock.last_Δt.mlir_data = step_Δt.mlir_data
    clock.last_stage_Δt.mlir_data = stage_Δt.mlir_data
    return nothing
end

end # module
