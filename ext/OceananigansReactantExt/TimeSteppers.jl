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
    calculate_pressure_correction!,
    correct_velocities_and_cache_previous_tendencies!,
    step_lagrangian_particles!,
    QuasiAdamsBashforth2TimeStepper

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    step_free_surface!,
    local_ab2_step!,
    compute_free_surface_tendency!

import Oceananigans.TimeSteppers: Clock, unit_time, time_step!, ab2_step!
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

    # Note: Δt cannot change
    if model.clock.last_Δt isa Reactant.TracedRNumber
        model.clock.last_Δt.mlir_data = Δt.mlir_data
    else
        model.clock.last_Δt = Δt
    end

    #=
    # Be paranoid and update state at iteration 0
    @trace if model.clock.iteration == 0
        update_state!(model, callbacks; compute_tendencies=true)
    end

    # Take an euler step if:
    #   * We detect that the time-step size has changed.
    #   * We detect that this is the "first" time-step, which means we
    #     need to take an euler step. Note that model.clock.last_Δt is
    #     initialized as Inf
    #   * The user has passed euler=true to time_step!
    @trace if Δt != model.clock.last_Δt
        euler = true
    end
    =#

    # If euler, then set χ = -0.5
    minus_point_five = convert(FT, -0.5)
    ab2_timestepper = model.timestepper
    χ = ifelse(euler, minus_point_five, ab2_timestepper.χ)
    χ₀ = ab2_timestepper.χ # Save initial value
    ab2_timestepper.χ = χ

    # Full step for tracers, fractional step for velocities.
    ab2_step!(model, Δt)

    tick!(model.clock, Δt)

    if model.clock.last_Δt isa Reactant.TracedRNumber
        model.clock.last_Δt.mlir_data = Δt.mlir_data
    else
        model.clock.last_Δt = Δt
    end

    # just one stage
    if model.clock.last_stage_Δt isa Reactant.TracedRNumber
        model.clock.last_stage_Δt.mlir_data = Δt.mlir_data
    else
        model.clock.last_stage_Δt = Δt
    end

    calculate_pressure_correction!(model, Δt)
    correct_velocities_and_cache_previous_tendencies!(model, Δt)

    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, Δt)

    # Return χ to initial value
    ab2_timestepper.χ = χ₀

    return nothing
end

end # module

