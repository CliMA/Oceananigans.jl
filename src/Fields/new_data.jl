using Oceananigans.Grids: total_length, topology

using OffsetArrays: OffsetArray

#####
##### Creating offset arrays for field data by dispatching on architecture.
#####

"""
Return a range of indices for a field located at `Cell` centers
`along a grid dimension of length `N` and with halo points `H`.
"""
offset_indices(loc, topo, N, H=0) = 1 - H : N + H

"""
Return a range of indices for a field located at cell `Face`s
`along a grid dimension of length `N` and with halo points `H`.
"""
offset_indices(::Type{Face}, ::Type{Bounded}, N, H=0) = 1 - H : N + H + 1

"""
Return a range of indices for a field along a 'reduced' dimension.
"""
offset_indices(::Type{Nothing}, topo, N, H=0) = 1 : 1

"""
    offset_underlying_data(underlying_data, grid::AbstractGrid, loc)

Returns an `OffsetArray` that maps to `underlying_data` in memory,
with offset indices appropriate for the `data` of a field on
a `grid` of `size(grid)` and located at `loc`.
"""
function offset_underlying_data(underlying_data, grid::AbstractGrid, loc)
    ii = offset_indices(loc[1], topology(grid, 1), grid.Nx, grid.Hx)
    jj = offset_indices(loc[2], topology(grid, 2), grid.Ny, grid.Hy)
    kk = offset_indices(loc[3], topology(grid, 3), grid.Nz, grid.Hz)

    return OffsetArray(underlying_data, ii, jj, kk)
end

"""
    new_data([FT=Float64], ::CPU, grid, loc)

Returns an `OffsetArray` of zeros of float type `FT`, with
parent data in CPU memory and indices corresponding to a field on a
`grid` of `size(grid)` and located at `loc`.
"""
function new_data(FT, ::CPU, grid, loc)
    underlying_data = zeros(FT, total_length(loc[1], topology(grid, 1), grid.Nx, grid.Hx),
                                total_length(loc[2], topology(grid, 2), grid.Ny, grid.Hy),
                                total_length(loc[3], topology(grid, 3), grid.Nz, grid.Hz))

    return offset_underlying_data(underlying_data, grid, loc)
end

"""
    new_data([FT=Float64], ::GPU, grid, loc)

Returns an `OffsetArray` of zeros of float type `FT`, with
parent data in GPU memory and indices corresponding to a field on a `grid`
of `size(grid)` and located at `loc`.
"""
function new_data(FT, ::GPU, grid, loc)
    underlying_data = CuArray{FT}(undef, total_length(loc[1], topology(grid, 1), grid.Nx, grid.Hx),
                                         total_length(loc[2], topology(grid, 2), grid.Ny, grid.Hy),
                                         total_length(loc[3], topology(grid, 3), grid.Nz, grid.Hz))

    underlying_data .= 0 # Ensure data is initially 0.

    return offset_underlying_data(underlying_data, grid, loc)
end

# Default to type of Grid
new_data(arch, grid, loc) = new_data(eltype(grid), arch, grid, loc)
