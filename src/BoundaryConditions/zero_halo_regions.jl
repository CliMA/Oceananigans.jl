using Oceananigans.Grids: underlying_left_halo_indices, underlying_right_halo_indices, topology
using Oceananigans: location

"""
    zero_halo_regions!(fields::Union{Tuple, NamedTuple})

Zero out the halo regions of each field in `fields`.
"""
function zero_halo_regions!(fields::Union{Tuple, NamedTuple}; kwargs...)

    for field in fields
        zero_halo_regions!(field; kwargs...)
    end

    return nothing
end

"""
    zero_halo_regions!(field)

Zero out the halo regions of `field`.
"""
zero_halo_regions!(field; dims=(1, 2, 3)) = zero_halo_regions!(parent(field), location(field), field.grid, dims=dims)

"""
    zero_halo_regions!(underlying_data, loc, grid)

Zero out the halo regions of the `underlying_data` of a field
at `loc`ation on `grid`.
"""
function zero_halo_regions!(underlying_data, loc, grid; dims=(1, 2, 3))

    dims = tuple(dims...)
    Lx, Ly, Lz = loc
    Tx, Ty, Tz = topology(grid)

    if 1 ∈ dims
        zero_west_halo!(underlying_data,  underlying_left_halo_indices(Lx, Tx, grid.Nx, grid.Hx))
        zero_east_halo!(underlying_data, underlying_right_halo_indices(Lx, Tx, grid.Nx, grid.Hx))
    end

    if 2 ∈ dims
        zero_south_halo!(underlying_data,  underlying_left_halo_indices(Ly, Ty, grid.Ny, grid.Hy))
        zero_north_halo!(underlying_data, underlying_right_halo_indices(Ly, Ty, grid.Ny, grid.Hy))
    end

    if 3 ∈ dims
           zero_top_halo!(underlying_data,  underlying_left_halo_indices(Lz, Tz, grid.Nz, grid.Hz))
        zero_bottom_halo!(underlying_data, underlying_right_halo_indices(Lz, Tz, grid.Nz, grid.Hz))
    end

    return nothing
end

  @inline zero_west_halo!(d, halo_indices) = @views @. d[halo_indices, :, :] = 0
 @inline zero_south_halo!(d, halo_indices) = @views @. d[:, halo_indices, :] = 0
@inline zero_bottom_halo!(d, halo_indices) = @views @. d[:, :, halo_indices] = 0

 @inline zero_east_halo!(d, halo_indices) = @views @. d[halo_indices, :, :] = 0
@inline zero_north_halo!(d, halo_indices) = @views @. d[:, halo_indices, :] = 0
  @inline zero_top_halo!(d, halo_indices) = @views @. d[:, :, halo_indices] = 0
