module Models

import Oceananigans

using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Models.HydrostaticFreeSurfaceModels: initialize_free_surface!, HydrostaticFreeSurfaceModel

using ..TimeSteppers: ReactantModel
using ..Grids: ReactantGrid, ReactantImmersedBoundaryGrid
using ..Grids: ShardedGrid, ShardedDistributed

import Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: maybe_extend_halos, FixedSubstepNumber
import Oceananigans: initialize!
import Oceananigans.Models:
        initialization_update_state!,
        complete_communication_and_compute_buffer!,
        interior_tendency_kernel_parameters
import Oceananigans.TimeSteppers: maybe_initialize_state!

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
    Oceananigans.Models.HydrostaticFreeSurfaceModels.initialize_free_surface!(model.free_surface, model.grid, model.velocities)
    return nothing
end

# Skip initialization_update_state! during model construction.
# Operations like update_state!, fill_halo_regions!, and initialize_free_surface!
# invoke KA.Kernel{ReactantBackend} which requires a @compile context.
# These are instead performed inside first_time_step! (which runs within @compile).
function initialization_update_state!(model::ReactantHFSM)
    return nothing
end

# No-op for Reactant: the iteration == 0 check evaluates at trace time,
# causing a redundant update_state! to be compiled into every time_step!.
# Instead, first_time_step! handles initialization explicitly.
maybe_initialize_state!(model::ReactantHFSM, callbacks) = nothing

# Undo all the pipelining for a `ShardedDistributed` architecture
complete_communication_and_compute_buffer!(model, ::ShardedGrid, ::ShardedDistributed) = nothing
interior_tendency_kernel_parameters(::ShardedDistributed, grid) = :xyz

end # module
