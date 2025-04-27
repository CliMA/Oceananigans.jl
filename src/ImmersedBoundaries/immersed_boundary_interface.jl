"""
    immersed_cell(i, j, k, grid)

Return true if a `cell` is "completely" immersed, and thus
is not part of the prognostic state.
"""
@inline immersed_cell(i, j, k, grid) = false

# Unpack to make defining new immersed boundaries more convenient
@inline immersed_cell(i, j, k, grid::ImmersedBoundaryGrid) =
    immersed_cell(i, j, k, grid.underlying_grid, grid.immersed_boundary)

"""
    inactive_cell(i, j, k, grid::ImmersedBoundaryGrid)

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
