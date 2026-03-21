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

const ReactantHFSM{TS, E} = Union{
    HydrostaticFreeSurfaceModel{TS, E, <:ReactantState},
    HydrostaticFreeSurfaceModel{TS, E, <:Distributed{<:ReactantState}},
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

end # module
