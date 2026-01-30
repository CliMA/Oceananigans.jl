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
    step_lagrangian_particles!,
    QuasiAdamsBashforth2TimeStepper,
    SplitRungeKuttaTimeStepper,
    cache_previous_tendencies!,
    rk_substep!,
    cache_current_fields!

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
    stage = 1
    last_Δt = convert(FT, Inf)
    last_stage_Δt = convert(FT, Inf)
    return Clock(; time=t, iteration=iter, stage, last_Δt, last_stage_Δt)
end

function Clock(grid::ShardedGrid)
    FT = Oceananigans.defaults.FloatType
    arch = architecture(grid)
    replicate = Sharding.Replicated(arch.connectivity)
    t = ConcreteRNumber(zero(FT), sharding=replicate)
    iter = ConcreteRNumber(0, sharding=replicate)
    stage = 1
    last_Δt = convert(FT, Inf)
    last_stage_Δt = convert(FT, Inf)
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
    ab2_step!(model, Δt, callbacks)
    cache_previous_tendencies!(model)

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

    update_state!(model, callbacks)
    step_lagrangian_particles!(model, Δt)

    # Return χ to initial value
    ab2_timestepper.χ = χ₀

    return nothing
end

function time_step!(model::ReactantModel{<:SplitRungeKuttaTimeStepper}, Δt; callbacks=[])
    # Eliminate problematic conditional, since our extension of first_time_step! ensures update_state! is called
    #=
    if model.clock.iteration == 0
        update_state!(model, callbacks)
    end
    =#

    cache_current_fields!(model)
    grid = model.grid

    ####
    #### Loop over the stages
    ####

    for (stage, β) in enumerate(model.timestepper.β)
        # Update the clock stage
        model.clock.stage = stage

        # Perform the substep
        Δτ = Δt / β
        rk_substep!(model, Δτ, callbacks)

        # Update the state
        update_state!(model, callbacks)
    end

    # Finalize step
    step_lagrangian_particles!(model, Δt)
    tick!(model.clock, Δt)

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
