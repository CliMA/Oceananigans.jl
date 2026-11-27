#####
##### Free surface nodes beneath an immersed top
#####

const GFBTIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedWithTop}
const ImmersedTopIBG = Union{GFBTIBG, PCBTIBG}

# The free surface is active in wet columns, including those beneath a top
@inline Grids.inactive_node(i, j, k, ibg::ImmersedTopIBG, ::Center, ::Center, ::Face) =
    ifelse(k == ibg.Nz + 1, static_column_depthᶜᶜᵃ(i, j, ibg) ≤ zero(ibg), inactive_cell(i, j, k, ibg) & inactive_cell(i, j, k-1, ibg))

# Reduce over y before z, like `(Face, Center, Face)`, so y-faces at the free surface see wet columns
@inline Grids.inactive_node(i, j, k, ibg::ImmersedTopIBG, ::Center, ::Face, ::Face) =
    inactive_node(i, j, k, ibg, c, c, f) & inactive_node(i, j-1, k, ibg, c, c, f)
