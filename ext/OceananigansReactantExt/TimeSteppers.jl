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
const ReactantClock = Clock{<:ConcreteRNumber}

function Base.setproperty!(clock::ReactantClock, prop::Symbol, value)
    if prop in (:last_Δt, :last_stage_Δt, :time, :iteration)
        clock_val = getproperty(clock, prop)
        if Reactant.Sharding.is_sharded(clock_val)
            sharding = clock_val.sharding
            converted_val = convert(innertype(clock_val), value)
            sharded_val = ConcreteRNumber(converted_val; sharding)
            return setfield!(clock, prop, sharded_val)
        end
    end
    return setfield!(clock, prop, value)
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

end # module
