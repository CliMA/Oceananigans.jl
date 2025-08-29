module TimeSteppers

using ..Architectures: ReactantState
using ..Grids: ShardedGrid, ReactantGrid

using Reactant
using Oceananigans

using Oceananigans: AbstractModel, Distributed
using Oceananigans.Grids: AbstractGrid, architecture
using Oceananigans.Utils: @apply_regionally, apply_regionally!
using Oceananigans.TimeSteppers:
    update_state!,
    tick!,
    compute_pressure_correction!,
    correct_velocities_and_cache_previous_tendencies!,
    step_lagrangian_particles!,
    QuasiAdamsBashforth2TimeStepper

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    step_free_surface!,
    compute_free_surface_tendency!

import Oceananigans.TimeSteppers: Clock, unit_time, first_time_step!, time_step!, ab2_step!
import Oceananigans: initialize!

const ReactantModel{TS} = Union{
    AbstractModel{TS, <:ReactantState},
    AbstractModel{TS, <:Distributed{<:ReactantState}}
}

function Clock(::ReactantGrid)
    FT = Oceananigans.defaults.FloatType
    t = ConcreteRNumber(zero(FT))
    iter = ConcreteRNumber(0)
    stage = 0 #ConcreteRNumber(0)
    last_Δt = zero(FT)
    last_stage_Δt = zero(FT)
    return Clock(; time=t, iteration=iter, stage, last_Δt, last_stage_Δt)
end

function Clock(grid::ShardedGrid)
    FT = Oceananigans.defaults.FloatType
    arch = architecture(grid)
    replicate = Sharding.Replicated(arch.connectivity)
    t = ConcreteRNumber(zero(FT), sharding=replicate)
    iter = ConcreteRNumber(0, sharding=replicate)
    stage = 0 #ConcreteRNumber(0)
    last_Δt = zero(FT)
    last_stage_Δt = zero(FT)
    return Clock(; time=t, iteration=iter, stage, last_Δt, last_stage_Δt)
end

function time_step!(model::ReactantModel{<:QuasiAdamsBashforth2TimeStepper{FT}}, Δt;
                    callbacks=[], euler=false) where FT

    update_state!(model, model.grid, callbacks; compute_tendencies=true)

    return nothing
end

function first_time_step!(model::ReactantModel, Δt)
    initialize!(model)
    # The first update_state is conditionally gated from within time_step! normally, but not Reactant
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

function first_time_step!(model::ReactantModel{<:QuasiAdamsBashforth2TimeStepper}, Δt)
    initialize!(model)
    # The first update_state is conditionally gated from within time_step! normally, but not Reactant
    update_state!(model)
    time_step!(model, Δt, euler=true)
    return nothing
end

end # module

