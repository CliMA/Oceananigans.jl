"""
$(TYPEDSIGNATURES)

Return true if a `cell` is "completely" immersed, and thus
is not part of the prognostic state.
"""
@inline immersed_cell(i, j, k, grid) = false

# Unpack to make defining new immersed boundaries more convenient
@inline immersed_cell(i, j, k, grid::ImmersedBoundaryGrid) =
    immersed_cell(i, j, k, grid.underlying_grid, grid.immersed_boundary)

"""
$(TYPEDSIGNATURES)

Return `true` if the tracer cell at `i, j, k` either (i) lies outside the `Bounded` domain
or (ii) lies within the immersed region of `ImmersedBoundaryGrid`.

Example
=======

Consider the configuration

```
   Immersed      Fluid
  =========== ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅

       c           c
      i-1          i

 | ========= |           |
 × === ∘ === ×     ∘     ×
 | ========= |           |

i-1          i
 f           f           f
```

We then have

* `inactive_node(i, 1, 1, grid, f, c, c) = false`

As well as

* `inactive_node(i,   1, 1, grid, c, c, c) = false`
* `inactive_node(i-1, 1, 1, grid, c, c, c) = true`
* `inactive_node(i-1, 1, 1, grid, f, c, c) = true`
"""
@inline inactive_cell(i, j, k, ibg::IBG) = immersed_cell(i, j, k, ibg) | inactive_cell(i, j, k, ibg.underlying_grid)
@inline inactive_cell(i::AbstractArray, j::AbstractArray, k::AbstractArray, ibg::IBG) = immersed_cell(i, j, k, ibg) .| inactive_cell(i, j, k, ibg.underlying_grid)

# Isolate periphery of the immersed boundary
@inline immersed_peripheral_node(i, j, k, ibg::IBG, LX, LY, LZ) =  peripheral_node(i, j, k, ibg, LX, LY, LZ) &
                                                                  !peripheral_node(i, j, k, ibg.underlying_grid, LX, LY, LZ)

@inline immersed_peripheral_node(i::AbstractArray, j::AbstractArray, k::AbstractArray, ibg::IBG, LX, LY, LZ) =  peripheral_node(i, j, k, ibg, LX, LY, LZ) .&
                                                                  Base.broadcast(!, peripheral_node(i, j, k, ibg.underlying_grid, LX, LY, LZ))

@inline immersed_inactive_node(i, j, k, ibg::IBG, LX, LY, LZ) = inactive_node(i, j, k, ibg, LX, LY, LZ) &
                                                                !inactive_node(i, j, k, ibg.underlying_grid, LX, LY, LZ)

@inline immersed_inactive_node(i::AbstractArray, j::AbstractArray, k::AbstractArray, ibg::IBG, LX, LY, LZ) =  inactive_node(i, j, k, ibg, LX, LY, LZ) .&
                                                                Base.broadcast(!, inactive_node(i, j, k, ibg.underlying_grid, LX, LY, LZ))

#####
##### Free surface nodes at the top of the water column
#####

"""
$(TYPEDSIGNATURES)

Return `true` if the water column at `(i, j)` holds no water. The free surface and the
barotropic transports exist only in wet columns, so they are masked by this rather than by
the activity of any individual cell.
"""
@inline dry_columnᶜᶜᵃ(i, j, ibg::IBG) = static_column_depthᶜᶜᵃ(i, j, ibg) ≤ zero(ibg)

# Flat directions have no halo, so their index collapses to 1 as in `immersed_cell`
@inline dry_columnᶜᶜᵃ(i, j, ibg::IBG{<:Any, Flat, <:Any}) = static_column_depthᶜᶜᵃ(1, j, ibg) ≤ zero(ibg)
@inline dry_columnᶜᶜᵃ(i, j, ibg::IBG{<:Any, <:Any, Flat}) = static_column_depthᶜᶜᵃ(i, 1, ibg) ≤ zero(ibg)
@inline dry_columnᶜᶜᵃ(i, j, ibg::IBG{<:Any, Flat, Flat})  = static_column_depthᶜᶜᵃ(1, 1, ibg) ≤ zero(ibg)

"""
$(TYPEDSIGNATURES)

Return the sum of `Δr` over the active cells of the face column at `(i, j)`. Unlike the minimum
of the neighbouring center depths, this holds when the bottom and the ceiling vary independently.
"""
@inline function active_column_depthᶠᶜᵃ(i, j, ibg::IBG)
    H = zero(ibg)
    for k in 1:ibg.Nz
        H += ifelse(peripheral_node(i, j, k, ibg, Face(), Center(), Center()), zero(ibg), Δrᶠᶜᶜ(i, j, k, ibg))
    end
    return H
end

"""
$(TYPEDSIGNATURES)

`active_column_depthᶠᶜᵃ`'s counterpart at `(Center, Face)` columns.
"""
@inline function active_column_depthᶜᶠᵃ(i, j, ibg::IBG)
    H = zero(ibg)
    for k in 1:ibg.Nz
        H += ifelse(peripheral_node(i, j, k, ibg, Center(), Face(), Center()), zero(ibg), Δrᶜᶠᶜ(i, j, k, ibg))
    end
    return H
end

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

`immersed_inactive_node`'s column-wise counterpart: `true` where the immersed boundary, and not the
underlying grid, makes the `(LX, LY)` column node dry.
"""
@inline immersed_dry_column_node(i, j, k, ibg::IBG, LX, LY) =  dry_column_node(i, j, ibg, LX, LY) &
                                                              !inactive_node(i, j, k, ibg.underlying_grid, LX, LY, Center())

"""
$(TYPEDSIGNATURES)

Return `inactive_node`, except at the topmost `Face` node in `z`, where the column is consulted
instead of the cell below.

The free surface lives at `k = Nz + 1` of every wet column, including columns capped by an ice
shelf, where `inactive_node` is `true` and would zero every consumer of `∇η`. This anchors the
free surface to the top of the water column.
"""
@inline top_inactive_node(i, j, k, grid, LX, LY, LZ) = inactive_node(i, j, k, grid, LX, LY, LZ)

@inline function top_inactive_node(i, j, k, ibg::IBG, ::Center, ::Center, ::Face)
    # A guard clause, not `ifelse`: `ifelse` would evaluate `dry_columnᶜᶜᵃ` in every column.
    k == ibg.Nz + 1 && return dry_columnᶜᶜᵃ(i, j, ibg)
    return inactive_node(i, j, k, ibg, Center(), Center(), Face())
end

@inline top_inactive_node(i, j, k, ibg::IBG, ::Face, LY, ::Face) = top_inactive_node(i, j, k, ibg, Center(), LY, Face()) &
                                                                   top_inactive_node(i-1, j, k, ibg, Center(), LY, Face())

@inline top_inactive_node(i, j, k, ibg::IBG, ::Center, ::Face, ::Face) = top_inactive_node(i, j, k, ibg, Center(), Center(), Face()) &
                                                                         top_inactive_node(i, j-1, k, ibg, Center(), Center(), Face())

"""
$(TYPEDSIGNATURES)

`immersed_inactive_node` built on [`top_inactive_node`](@ref), so that the free surface of a wet
column is never mistaken for an immersed node.
"""
@inline immersed_top_inactive_node(i, j, k, ibg::IBG, LX, LY, LZ) = top_inactive_node(i, j, k, ibg, LX, LY, LZ) &
                                                                    !top_inactive_node(i, j, k, ibg.underlying_grid, LX, LY, LZ)
