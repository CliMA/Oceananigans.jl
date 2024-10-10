using Oceananigans.Grids: AbstractVerticalCoordinateUnderlyingGrid, ZStarUnderlyingGrid

import Oceananigans.Grids: retrieve_static_grid
import Oceananigans.Grids: reference_zspacings, reference_znodes, vertical_scaling, previous_vertical_scaling

const ImmersedAbstractVerticalCoordinateGrid = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractVerticalCoordinateUnderlyingGrid}
const ImmersedZStarGrid = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarUnderlyingGrid}

reference_znodes(grid::ImmersedBoundaryGrid, ℓz) = reference_znodes(grid.underlying_grid)
reference_zspacings(grid::ImmersedBoundaryGrid, ℓz) = reference_zspacings(grid.underlying_grid)

@inline vertical_scaling(i, j, k, grid::ImmersedBoundaryGrid, ℓx, ℓy, ℓz) = vertical_scaling(i, j, k, grid.underlying_grid, ℓx, ℓy, ℓz)
@inline previous_vertical_scaling(i, j, k, grid::ImmersedBoundaryGrid, ℓx, ℓy, ℓz) = previous_vertical_scaling(i, j, k, grid.underlying_grid, ℓx, ℓy, ℓz)

function retrieve_static_grid(ib::ImmersedAbstractVerticalCoordinateGrid) 
    immersed_boundary = ib.immersed_boundary
    active_cells_map  = !isnothing(ib.active_cells_map)
    underlying_grid   = retrieve_static_grid(ib.underlying_grid)

    return ImmersedBoundaryGrid(underlying_grid, immersed_boundary; active_cells_map)
end