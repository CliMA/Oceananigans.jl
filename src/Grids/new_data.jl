using Oceananigans.Grids: total_length, topology

using OffsetArrays: OffsetArray

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

"""
Return a range of indices for a field along a 'reduced' dimension.
"""
offset_indices(::Type{Nothing}, topo, N, H=0) = 1 : 1

offset_indices(L, T, N, H, ::Colon) = offset_indices(L, T, N, H)
offset_indices(L, T, N, H, i::UnitRange) = i

function offset_data(underlying_data::AbstractArray{FT, 3}, loc, topo, N, H, indices) where FT
    ii, jj, kk = offset_indices.(loc, topo, N, H, indices)
    return OffsetArray(underlying_data, ii, jj, kk)
end

function offset_data(underlying_data::AbstractArray{FT, 2}, loc, topo, N, H, indices) where FT
    ii, jj, kk = offset_indices.(loc, topo, N, H, indices)
    return OffsetArray(underlying_data, ii, jj)
end

"""
    offset_data(underlying_data, grid::AbstractGrid, loc)

Returns an `OffsetArray` that maps to `underlying_data` in memory,
with offset indices appropriate for the `data` of a field on
a `grid` of `size(grid)` and located at `loc`.
"""
offset_data(underlying_data, grid::AbstractGrid, loc, indices) =
    offset_data(underlying_data, loc, topology(grid), size(grid), halo_size(grid), indices)

"""
    new_data(FT, grid, loc, indices)

Returns an `OffsetArray` of zeros of float type `FT` on `arch`itecture,
with indices corresponding to a field on a `grid` of `size(grid)` and located at `loc`.
"""
function new_data(FT::DataType, grid, loc, indices)
    arch = architecture(grid)
    Tx, Ty, Tz = total_size(loc, grid, indices)
    underlying_data = zeros(FT, arch, Tx, Ty, Tz)
    return offset_data(underlying_data, grid, loc, indices)
end

new_data(grid, loc, indices) = new_data(eltype(grid), grid, loc, indices)

