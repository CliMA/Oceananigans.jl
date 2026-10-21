
const c = Center()
const f = Face()

using ReactantCore: ReactantCore

# The first and last indices with data next to a side: up to the boundary face, and beyond it the exterior Center node of an Open side
@inline first_index_with_data(::Closed, ℓ)      = 1
@inline first_index_with_data(::Open, ::Center) = 0
@inline first_index_with_data(::Open, ::Face)   = 1

@inline last_index_with_data(::Closed, ::Center, N) = N
@inline last_index_with_data(::Closed, ::Face, N)   = N + 1
@inline last_index_with_data(::Open, ℓ, N)          = N + 1

# Whether a stencil from `first_point` to `last_point` reads only what the sides of a direction let it read; only Closed and Open sides bound it
@inline within_bounds(first_point, last_point, topology, ℓ, N) = true
@inline within_bounds(first_point, last_point, ::Type{<:Sides{L, R}}, ℓ, N) where {L <: BoundarySide, R <: BoundarySide} = (first_point >= first_index_with_data(L(), ℓ)) & (last_point <= last_index_with_data(R(), ℓ, N))
@inline within_bounds(first_point, last_point, ::Type{<:Sides{L}}, ℓ, N) where L <: BoundarySide         = first_point >= first_index_with_data(L(), ℓ)
@inline within_bounds(first_point, last_point, ::Type{<:Sides{<:Any, R}}, ℓ, N) where R <: BoundarySide  = last_point  <= last_index_with_data(R(), ℓ, N)

#####
##### Exterior node and peripheral node
#####

"""
$(TYPEDSIGNATURES)

Return `true` when the tracer cell at `i, j, k` is "external" to the domain boundary.

`inactive_cell`s include the halo cells beyond a `Closed` side, the halo cells beyond the exterior cell of an `Open` side,
and cells within an immersed boundary.
Cells that are staggered with respect to tracer cells and which lie _on_ the boundary are considered active.
"""
@inline inactive_cell(i, j, k, grid) = !(within_bounds(i, i, topology(grid, 1), c, grid.Nx) &
                                         within_bounds(j, j, topology(grid, 2), c, grid.Ny) &
                                         within_bounds(k, k, topology(grid, 3), c, grid.Nz))

@inline function inactive_cell(i::AbstractArray, j::AbstractArray, k::AbstractArray, grid)
    iᵐ = ReactantCore.materialize_traced_array(i)
    jᵐ = ReactantCore.materialize_traced_array(j)
    kᵐ = ReactantCore.materialize_traced_array(k)
    return .!(within_bounds.(iᵐ, iᵐ, topology(grid, 1), Ref(c), grid.Nx) .&
              within_bounds.(jᵐ, jᵐ, topology(grid, 2), Ref(c), grid.Ny) .&
              within_bounds.(kᵐ, kᵐ, topology(grid, 3), Ref(c), grid.Nz))
end

@inline active_cell(i, j, k, grid) = !inactive_cell(i, j, k, grid)

"""
$(TYPEDSIGNATURES)

Return `true` when the location `(LX, LY, LZ)` is "inactive" and thus not directly
associated with an "active" cell.

For `Face` locations, this means the node is surrounded by `inactive_cell`s:
the interfaces of "active" cells are _not_ `inactive_node`.

For `Center` locations, this means the direction is `Bounded` and that the
cell or interface centered on the location is completely outside the active
region of the grid.
"""
@inline inactive_node(i, j, k, grid, LX, LY, LZ) = inactive_cell(i, j, k, grid)

@inline inactive_node(i, j, k, grid, ::Face, LY, LZ) = inactive_cell(i, j, k, grid) & inactive_cell(i-1, j, k, grid)
@inline inactive_node(i, j, k, grid, LX, ::Face, LZ) = inactive_cell(i, j, k, grid) & inactive_cell(i, j-1, k, grid)
@inline inactive_node(i, j, k, grid, LX, LY, ::Face) = inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)

@inline inactive_node(i, j, k, grid, ::Face, ::Face, LZ) = inactive_node(i, j, k, grid, c, f, c) & inactive_node(i-1, j, k, grid, c, f, c)
@inline inactive_node(i, j, k, grid, ::Face, LY, ::Face) = inactive_node(i, j, k, grid, c, c, f) & inactive_node(i-1, j, k, grid, c, c, f)
@inline inactive_node(i, j, k, grid, LX, ::Face, ::Face) = inactive_node(i, j, k, grid, c, f, c) & inactive_node(i, j, k-1, grid, c, f, c)

@inline inactive_node(i, j, k, grid, ::Face, ::Face, ::Face) = inactive_node(i, j, k, grid, c, f, f) & inactive_node(i-1, j, k, grid, c, f, f)

# Fallback for non-immersed grids, ImmersedBoundaries extends this for `ImmersedBoundaryGrid`.
@inline immersed_inactive_node(i, j, k, grid, LX, LY, LZ) = false

@inline unit_vector(::Val{1}) = (1, 0, 0)
@inline unit_vector(::Val{2}) = (0, 1, 0)
@inline unit_vector(::Val{3}) = (0, 0, 1)

# A node has data within the sides of the direction d and outside the immersed boundary
@inline function node_has_data(i, j, k, grid, ::Val{d}, ℓx, ℓy, ℓz) where d
    n = (i, j, k)[d]
    return within_bounds(n, n, topology(grid, d), (ℓx, ℓy, ℓz)[d], size(grid, d)) & !immersed_inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)
end

"""
$(TYPEDSIGNATURES)

The opposite of inactive_node(args...).
"""
@inline active_node(args...) = !inactive_node(args...)

"""
$(TYPEDSIGNATURES)

Return `true` when the location `(LX, LY, LZ)`, is _either_ inactive or
lies on the boundary between inactive and active cells in a `Bounded` direction.
"""
@inline peripheral_node(i, j, k, grid, LX, LY, LZ) = inactive_cell(i, j, k, grid)

@inline peripheral_node(i, j, k, grid, ::Face, LY, LZ) = inactive_cell(i, j, k, grid) | inactive_cell(i-1, j, k, grid)
@inline peripheral_node(i, j, k, grid, LX, ::Face, LZ) = inactive_cell(i, j, k, grid) | inactive_cell(i, j-1, k, grid)
@inline peripheral_node(i, j, k, grid, LX, LY, ::Face) = inactive_cell(i, j, k, grid) | inactive_cell(i, j, k-1, grid)

@inline peripheral_node(i, j, k, grid, ::Face, ::Face, LZ) = peripheral_node(i, j, k, grid, c, f, c) | peripheral_node(i-1, j, k, grid, c, f, c)
@inline peripheral_node(i, j, k, grid, ::Face, LY, ::Face) = peripheral_node(i, j, k, grid, c, c, f) | peripheral_node(i-1, j, k, grid, c, c, f)
@inline peripheral_node(i, j, k, grid, LX, ::Face, ::Face) = peripheral_node(i, j, k, grid, c, f, c) | peripheral_node(i, j, k-1, grid, c, f, c)

@inline peripheral_node(i, j, k, grid, ::Face, ::Face, ::Face) = peripheral_node(i, j, k, grid, c, f, f) | peripheral_node(i-1, j, k, grid, c, f, f)

# Fallback for non-immersed grids, ImmersedBoundaries extends this for `ImmersedBoundaryGrid`.
@inline immersed_peripheral_node(i, j, k, grid, LX, LY, LZ) = false

"""
$(TYPEDSIGNATURES)

Return `true` when the location `(LX, LY, LZ)` lies on a boundary.
"""
@inline boundary_node(i, j, k, grid, LX, LY, LZ) = peripheral_node(i, j, k, grid, LX, LY, LZ) & !inactive_node(i, j, k, grid, LX, LY, LZ)

"""
$(TYPEDSIGNATURES)

Return `true` when the location `(LX, LY, LZ)` is the active cell just above the bottom.
"""
@inline bottommost_active_node(i, j, k, grid, LX, LY, LZ) = active_node(i, j, k, grid, LX, LY, LZ) & inactive_node(i, j, k - 1, grid, LX, LY, LZ)
