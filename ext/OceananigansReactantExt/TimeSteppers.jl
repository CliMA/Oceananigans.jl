module TimeSteppers

using ..Architectures: ReactantState
using ..Grids: ShardedGrid, ReactantGrid

using Reactant
using Oceananigans

using Oceananigans: AbstractModel, Distributed
using Oceananigans.Grids: architecture

import Oceananigans.TimeSteppers: Clock, first_time_step!, maybe_initialize_state!
import Oceananigans: initialize!

const ReactantModel{TS} = Union{
    AbstractModel{TS, <:ReactantState},
    AbstractModel{TS, <:Distributed{<:ReactantState}}
}

function Clock(::ReactantGrid)
    FT = Oceananigans.defaults.FloatType
    t = ConcreteRNumber(zero(FT))
    iter = ConcreteRNumber(0)
    stage = 1
    last_Δt = ConcreteRNumber(convert(FT, Inf))
    last_stage_Δt = ConcreteRNumber(convert(FT, Inf))
    return Clock(; time=t, iteration=iter, stage, last_Δt, last_stage_Δt)
end

function Clock(grid::ShardedGrid)
    FT = Oceananigans.defaults.FloatType
    arch = architecture(grid)
    replicate = Sharding.Replicated(arch.connectivity)
    t = ConcreteRNumber(zero(FT), sharding=replicate)
    iter = ConcreteRNumber(0, sharding=replicate)
    stage = 1
    last_Δt = ConcreteRNumber(convert(FT, Inf), sharding=replicate)
    last_stage_Δt = ConcreteRNumber(convert(FT, Inf), sharding=replicate)
    return Clock(; time=t, iteration=iter, stage, last_Δt, last_stage_Δt)
end

# Reactant handles initialization via first_time_step!, so this is a no-op.
maybe_initialize_state!(::ReactantModel, callbacks) = nothing

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
