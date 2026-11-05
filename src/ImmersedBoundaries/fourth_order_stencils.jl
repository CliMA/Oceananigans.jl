using Oceananigans.Operators: Operators

#####
##### Fourth-order-accurate first derivatives on immersed boundary grids
#####
##### The four-point stencil is used where it is used on the underlying grid, except in the vertical direction
##### of grids with partial cells, whose spacing is not regular next to the immersed boundary.
##### Stencils that include immersed (inactive) nodes fall back to second order via `inactive_node`.
#####

@inline Operators.fourth_order_x_stencil(ibg::IBG) = Operators.fourth_order_x_stencil(ibg.underlying_grid)
@inline Operators.fourth_order_y_stencil(ibg::IBG) = Operators.fourth_order_y_stencil(ibg.underlying_grid)
@inline Operators.fourth_order_z_stencil(ibg::IBG) = Operators.fourth_order_z_stencil(ibg.underlying_grid)

const PartialCellIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:PartialCellVerticalBoundaries}

@inline Operators.fourth_order_z_stencil(::PartialCellIBG) = Val(false)
