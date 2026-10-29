module Models

import Oceananigans

import Oceananigans.TimeSteppers: reconcile_state!, maybe_prepare_first_time_step!
using Oceananigans.TimeSteppers: RungeKutta3TimeStepper, SplitRungeKuttaTimeStepper
import Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: maybe_extend_halos, FixedSubstepNumber
import Oceananigans: initialize!

using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using Oceananigans.Models.NonhydrostaticModels: NonhydrostaticModel
using Oceananigans.Models.ShallowWaterModels: ShallowWaterModel
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: initialize_closure_fields!

using ..TimeSteppers: ReactantModel
import ..TimeSteppers: initialize_state!
using ..Grids: ShardedGrid, ShardedDistributed

import Oceananigans.Models:
        complete_communication_and_compute_buffer!,
        interior_tendency_kernel_parameters
import Oceananigans.Advection: default_weno_weight_computation


const ReactantHFSM{TS, E} = Union{
    HydrostaticFreeSurfaceModel{TS, E, <:ReactantState},
    HydrostaticFreeSurfaceModel{TS, E, <:ShardedDistributed},
}

const ReactantNonhydrostaticModel{TS, E} = Union{
    NonhydrostaticModel{TS, E, <:ReactantState},
    NonhydrostaticModel{TS, E, <:ShardedDistributed},
}

const ReactantShallowWaterModel = Union{
    ShallowWaterModel{<:Any, <:ReactantState},
    ShallowWaterModel{<:Any, <:ShardedDistributed},
}

const OceananigansReactantModel = Union{ReactantHFSM, ReactantNonhydrostaticModel, ReactantShallowWaterModel}

# On ReactantState construction runs no kernels (`initialize_on_construction!`), so `initialize!` also
# runs the `update_state!` that construction runs on other architectures. Idempotent.
initialize!(model::ReactantHFSM) = initialize_state!(model)
initialize!(model::ReactantNonhydrostaticModel) = initialize_state!(model)
initialize!(model::ReactantShallowWaterModel) = initialize_state!(model)

function initialize_state!(model::OceananigansReactantModel, callbacks=[])
    prepare_initial_state!(model)
    update_state!(model, callbacks)
    return nothing
end

prepare_initial_state!(model::ReactantHFSM) = reconcile_state!(model)
prepare_initial_state!(model::ReactantNonhydrostaticModel) = initialize_closure_fields!(model.closure_fields, model.closure, model)
prepare_initial_state!(model::ReactantShallowWaterModel) = nothing

# No-op for Reactant: the iteration == 0 check evaluates at trace time,
# causing a redundant update_state! to be compiled into every time_step!.
# Instead, first_time_step! handles initialization explicitly.
maybe_prepare_first_time_step!(model::ReactantHFSM, Δt, callbacks) = nothing
maybe_prepare_first_time_step!(model::ReactantHFSM{<:RungeKutta3TimeStepper}, Δt, callbacks) = nothing
maybe_prepare_first_time_step!(model::ReactantHFSM{<:SplitRungeKuttaTimeStepper}, Δt, callbacks) = nothing

# Undo all the pipelining for a `ShardedDistributed` architecture
complete_communication_and_compute_buffer!(model, ::ShardedGrid, ::ShardedDistributed) = nothing
interior_tendency_kernel_parameters(::ShardedDistributed, grid) = :xyz

# Reactant uses CUDA version of the code to uplift program description to MLIR.
# Since default `weno_weight_computation` on CUDA uses LLVM's NVPTX intrinsics
# it causes Reactant to crash.
# We need to fall back to different optimization when running with Reactant
default_weno_weight_computation(::ReactantState) = Oceananigans.Utils.ConvertingDivision{Float32}

end # module
