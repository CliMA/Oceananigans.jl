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

function offset_data(underlying_data::AbstractArray{FT, 3}, loc, topo, N, H) where FT
    ii = offset_indices(loc[1], topo[1], N[1], H[1])
    jj = offset_indices(loc[2], topo[2], N[2], H[2])
    kk = offset_indices(loc[3], topo[3], N[3], H[3])

    return OffsetArray(underlying_data, ii, jj, kk)
end

function offset_data(underlying_data::AbstractArray{FT, 2}, loc, topo, N, H) where FT
    ii = offset_indices(loc[1], topo[1], N[1], H[1])
    jj = offset_indices(loc[2], topo[2], N[2], H[2])

    return OffsetArray(underlying_data, ii, jj)
end

"""
    offset_data(underlying_data, grid::AbstractGrid, loc)

Returns an `OffsetArray` that maps to `underlying_data` in memory,
with offset indices appropriate for the `data` of a field on
a `grid` of `size(grid)` and located at `loc`.
"""
offset_data(underlying_data, grid::AbstractGrid, loc) =
    offset_data(underlying_data, loc, topology(grid), size(grid), halo_size(grid))

"""
    new_data([FT=Float64], arch, grid, loc)

Returns an `OffsetArray` of zeros of float type `FT` on `arch`itecture,
with indices corresponding to a field on a `grid` of `size(grid)` and located at `loc`.
"""
function new_data(FT, arch, grid, loc)
    underlying_data = zeros(FT, arch,
                            total_length(loc[1], topology(grid, 1), grid.Nx, grid.Hx),
                            total_length(loc[2], topology(grid, 2), grid.Ny, grid.Hy),
                            total_length(loc[3], topology(grid, 3), grid.Nz, grid.Hz))

    return offset_data(underlying_data, grid, loc)
end

# Default to type of Grid
new_data(arch, grid, loc) = new_data(eltype(grid), arch, grid, loc)
