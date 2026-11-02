module Models

using ..Grids: ShardedDistributed, ShardedGrid
using ..TimeSteppers: ReactantModel
using Oceananigans: Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using Oceananigans.TimeSteppers: RungeKutta3TimeStepper, SplitRungeKuttaTimeStepper, reconcile_state!

const ReactantHFSM{TS, E} = Union{
    HydrostaticFreeSurfaceModel{TS, E, <:ReactantState},
    HydrostaticFreeSurfaceModel{TS, E, <:ShardedDistributed},
}

function Oceananigans.initialize!(model::ReactantHFSM)
    reconcile_state!(model)
    return nothing
end

# No-op for Reactant: the iteration == 0 check evaluates at trace time,
# causing a redundant update_state! to be compiled into every time_step!.
# Instead, first_time_step! handles initialization explicitly.
Oceananigans.TimeSteppers.maybe_prepare_first_time_step!(model::ReactantHFSM, Δt, callbacks) = nothing
Oceananigans.TimeSteppers.maybe_prepare_first_time_step!(model::ReactantHFSM{<:RungeKutta3TimeStepper}, Δt, callbacks) = nothing
Oceananigans.TimeSteppers.maybe_prepare_first_time_step!(model::ReactantHFSM{<:SplitRungeKuttaTimeStepper}, Δt, callbacks) = nothing

# Undo all the pipelining for a `ShardedDistributed` architecture
Oceananigans.Models.complete_communication_and_compute_buffer!(model, ::ShardedGrid, ::ShardedDistributed) = nothing
Oceananigans.Models.interior_tendency_kernel_parameters(::ShardedDistributed, grid) = :xyz

# Reactant uses CUDA version of the code to uplift program description to MLIR.
# Since default `weno_weight_computation` on CUDA uses LLVM's NVPTX intrinsics
# it causes Reactant to crash.
# We need to fall back to different optimization when running with Reactant
Oceananigans.Advection.default_weno_weight_computation(::ReactantState) = Oceananigans.Utils.ConvertingDivision{Float32}

end # module
