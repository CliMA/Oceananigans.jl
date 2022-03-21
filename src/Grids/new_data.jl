using Oceananigans.Grids: total_length, topology

using OffsetArrays: OffsetArray

default_indices(n) = Tuple(Colon() for i=1:n)

#####
##### Creating offset arrays for field data by dispatching on architecture.
#####

"""
Return a range of indices for a field located at either cell `Center`s or `Face`s along a
grid dimension which is `Periodic`, or cell `Center`s for a grid dimension which is `Bounded`.
The dimension has length `N` and `H` halo points.
"""
offset_indices(loc, topo, N, H=0) = 1 - H : N + H

"""
Return a range of indices for a field located at cell `Face`s along a grid dimension which
is `Bounded` and has length `N` and with halo points `H`.
"""
offset_indices(::Type{Face}, ::Type{Bounded}, N, H=0) = 1 - H : N + H + 1
offset_indices(::Type{Face}, ::Type{RightConnected}, N, H=0) = 1 - H : N + H + 1
offset_indices(::Type{Face}, ::Type{LeftConnected}, N, H=0) = 1 - H : N + H + 1

"""
Return a range of indices for a field along a 'reduced' dimension.
"""
offset_indices(::Type{Nothing}, topo, N, H=0) = 1 : 1

offset_indices(L, T, N, H, ::Colon) = offset_indices(L, T, N, H)
offset_indices(L, T, N, H, r::UnitRange) = r
offset_indices(::Type{Nothing}, T, N, H, ::UnitRange) = 1:1

function offset_data(underlying_data::AbstractArray, loc, topo, N, H, indices=default_indices(length(loc)))
    ii = offset_indices.(loc, topo, N, H, indices)
    # Add extra indices for arrays of higher dimension than loc, topo, etc.
    extra_ii = Tuple(axes(underlying_data, d) for d in length(ii)+1:ndims(underlying_data))
    return OffsetArray(underlying_data, ii..., extra_ii...)
end

"""
    offset_data(underlying_data, grid::AbstractGrid, loc)

Returns an `OffsetArray` that maps to `underlying_data` in memory,
with offset indices appropriate for the `data` of a field on
a `grid` of `size(grid)` and located at `loc`.
"""
offset_data(underlying_data::AbstractArray, grid::AbstractGrid, loc, indices=default_indices(length(loc))) =
    offset_data(underlying_data, loc, topology(grid), size(grid), halo_size(grid), indices)

"""
    new_data(FT, grid, loc, indices)

Returns an `OffsetArray` of zeros of float type `FT` on `arch`itecture,
with indices corresponding to a field on a `grid` of `size(grid)` and located at `loc`.
"""
function new_data(FT::DataType, grid::AbstractGrid, loc, indices=default_indices(length(loc)))
    arch = architecture(grid)
    Tx, Ty, Tz = total_size(loc, grid, indices)
    underlying_data = zeros(FT, arch, Tx, Ty, Tz)
    return offset_data(underlying_data, grid, loc, indices)
end

new_data(grid, loc, indices=default_indices) = new_data(eltype(grid), grid, loc, indices)

