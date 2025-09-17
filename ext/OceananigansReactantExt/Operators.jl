module Operators

import Oceananigans
import Oceananigans: Face, Center
import Oceananigans.Operators: not_peripheral_node
import Oceananigans.Grids: active_cell

using ..Grids: ReactantImmersedBoundaryGrid

@inline not_peripheral_node(i, j, k, grid::ReactantImmersedBoundaryGrid, LX, LY, LZ) = active_cell(i, j, k, grid)

@inline not_peripheral_node(i, j, k, grid::ReactantImmersedBoundaryGrid, ::Face, LY, LZ) = active_cell(i, j, k, grid) & active_cell(i-1, j, k, grid)
@inline not_peripheral_node(i, j, k, grid::ReactantImmersedBoundaryGrid, LX, ::Face, LZ) = active_cell(i, j, k, grid) & active_cell(i, j-1, k, grid)
@inline not_peripheral_node(i, j, k, grid::ReactantImmersedBoundaryGrid, LX, LY, ::Face) = active_cell(i, j, k, grid) & active_cell(i, j, k-1, grid)

end # module