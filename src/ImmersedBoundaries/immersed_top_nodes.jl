#####
##### Free surface nodes beneath an immersed top
#####

const ImmersedTopIBG = Union{GFBTIBG, PCBTIBG}

"""
$(TYPEDSIGNATURES)

Return `true` if the water column at `(i, j)` holds no water. The free surface and the
barotropic transports exist only in wet columns, so beneath an immersed top they are masked
by this rather than by the activity of the cell at `k = Nz`.
"""
@inline dry_columnᶜᶜᵃ(i, j, ibg::IBG) = static_column_depthᶜᶜᵃ(i, j, ibg) ≤ zero(ibg)

# Flat directions have no halo, so their index collapses to 1 as in `immersed_cell`
@inline dry_columnᶜᶜᵃ(i, j, ibg::IBG{<:Any, Flat, <:Any}) = static_column_depthᶜᶜᵃ(1, j, ibg) ≤ zero(ibg)
@inline dry_columnᶜᶜᵃ(i, j, ibg::IBG{<:Any, <:Any, Flat}) = static_column_depthᶜᶜᵃ(i, 1, ibg) ≤ zero(ibg)
@inline dry_columnᶜᶜᵃ(i, j, ibg::IBG{<:Any, Flat, Flat})  = static_column_depthᶜᶜᵃ(1, 1, ibg) ≤ zero(ibg)

"""
$(TYPEDSIGNATURES)

Return `true` if every column touching the `(LX, LY)` node is dry, composed across a `Face`
as `inactive_node` composes `inactive_cell`.
"""
@inline dry_column_node(i, j, ibg::IBG, ::Center, ::Center) = dry_columnᶜᶜᵃ(i, j, ibg)
@inline dry_column_node(i, j, ibg::IBG, ::Face,   LY)       = dry_column_node(i, j, ibg, Center(), LY) & dry_column_node(i-1, j, ibg, Center(), LY)
@inline dry_column_node(i, j, ibg::IBG, ::Center, ::Face)   = dry_column_node(i, j, ibg, Center(), Center()) & dry_column_node(i, j-1, ibg, Center(), Center())

"""
$(TYPEDSIGNATURES)

Column-wise counterpart of `immersed_inactive_node`: `true` where the immersed boundary, and not the
underlying grid, makes the `(LX, LY)` column node dry.
"""
@inline immersed_dry_column_node(i, j, k, ibg::IBG, LX, LY) =  dry_column_node(i, j, ibg, LX, LY) &
                                                              !inactive_node(i, j, k, ibg.underlying_grid, LX, LY, Center())

"""
$(TYPEDSIGNATURES)

Return `inactive_node`, except beneath an immersed top at the `Face` node `k = Nz + 1`, where
the column is consulted instead of the cell below.

The free surface lives at `k = Nz + 1` of every wet column, including columns capped by an
immersed top, where `inactive_node` is `true` and would zero every consumer of `∇η`.
"""
@inline top_inactive_node(i, j, k, grid, LX, LY, LZ) = inactive_node(i, j, k, grid, LX, LY, LZ)

@inline function top_inactive_node(i, j, k, ibg::ImmersedTopIBG, ::Center, ::Center, ::Face)
    # a guard clause, not `ifelse`, so `dry_columnᶜᶜᵃ` is evaluated only at the free surface
    k == ibg.Nz + 1 && return dry_columnᶜᶜᵃ(i, j, ibg)
    return inactive_node(i, j, k, ibg, Center(), Center(), Face())
end

@inline top_inactive_node(i, j, k, ibg::ImmersedTopIBG, ::Face, LY, ::Face) = top_inactive_node(i,   j, k, ibg, Center(), LY, Face()) &
                                                                             top_inactive_node(i-1, j, k, ibg, Center(), LY, Face())

@inline top_inactive_node(i, j, k, ibg::ImmersedTopIBG, ::Center, ::Face, ::Face) = top_inactive_node(i, j,   k, ibg, Center(), Center(), Face()) &
                                                                                   top_inactive_node(i, j-1, k, ibg, Center(), Center(), Face())

"""
$(TYPEDSIGNATURES)

`immersed_inactive_node` built on [`top_inactive_node`](@ref), so that the free surface of a wet
column is never mistaken for an immersed node.
"""
@inline immersed_top_inactive_node(i, j, k, ibg::IBG, LX, LY, LZ) =  top_inactive_node(i, j, k, ibg, LX, LY, LZ) &
                                                                    !top_inactive_node(i, j, k, ibg.underlying_grid, LX, LY, LZ)
