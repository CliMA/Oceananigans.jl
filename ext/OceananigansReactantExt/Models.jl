module Models

import Oceananigans

import Oceananigans.TimeSteppers: reconcile_state!, maybe_prepare_first_time_step!
import Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: maybe_extend_halos, FixedSubstepNumber
import Oceananigans: initialize!

using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel

using ..TimeSteppers: ReactantModel
using ..Grids: ReactantGrid, ReactantImmersedBoundaryGrid
using ..Grids: ShardedGrid, ShardedDistributed

import Oceananigans.Models:
        complete_communication_and_compute_buffer!,
        interior_tendency_kernel_parameters

import Oceananigans.Advection: default_weno_weight_computation
using Oceananigans.Utils: ConvertingDivision

const ReactantHFSM{TS, E} = Union{
    HydrostaticFreeSurfaceModel{TS, E, <:ReactantState},
    HydrostaticFreeSurfaceModel{TS, E, <:ShardedDistributed},
}

initialize_immersed_boundary_grid!(grid) = nothing

using Oceananigans.ImmersedBoundaries:
    GridFittedBottom,
    PartialCellBottom

function initialize_immersed_boundary_grid!(ibg::ReactantImmersedBoundaryGrid)
    # TODO This assumes that the IBG is GridFittedBottom or PartialCellBottom
    needs_initialization = ibg.immersed_boundary isa GridFittedBottom ||
                           ibg.immersed_boundary isa PartialCellBottom

    if needs_initialization
        ib = ibg.immersed_boundary
        bottom_field = ib.bottom_height
        grid = ibg.underlying_grid
        Oceananigans.ImmersedBoundaries.compute_numerical_bottom_height!(bottom_field, grid, ib)
        Oceananigans.BoundaryConditions.fill_halo_regions!(bottom_field)
    end

    return nothing
end

function initialize!(model::ReactantHFSM)
    initialize_immersed_boundary_grid!(model.grid)
    reconcile_state!(model)
    return nothing
end

# No-op for Reactant: the iteration == 0 check evaluates at trace time,
# causing a redundant update_state! to be compiled into every time_step!.
# Instead, first_time_step! handles initialization explicitly.
maybe_prepare_first_time_step!(model::ReactantHFSM, callbacks) = nothing

# Undo all the pipelining for a `ShardedDistributed` architecture
complete_communication_and_compute_buffer!(model, ::ShardedGrid, ::ShardedDistributed) = nothing
interior_tendency_kernel_parameters(::ShardedDistributed, grid) = :xyz

# Reactant uses CUDA version of the code to uplift program description to MLIR.
# Since the default `BackendOptimizedDivision` uses LLVM's NVPTX intrinsics,
# Reactant cannot consume it and falls back to a Float32-converted Newton division.
default_weno_weight_computation(::ReactantState) = ConvertingDivision{Float32}

end # module
