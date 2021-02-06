using Oceananigans.Grids: left_halo_indices, right_halo_indices, underlying_left_halo_indices, underlying_right_halo_indices
using Oceananigans.Fields: AbstractField

west_halo(f::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    view(f.data, left_halo_indices(LX, topology(f, 1), f.grid.Nx, f.grid.Hx), :, :)

east_halo(f::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    view(f.data, right_halo_indices(LX, topology(f, 1), f.grid.Nx, f.grid.Hx), :, :)

south_halo(f::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    view(f.data, :, left_halo_indices(LY, topology(f, 2), f.grid.Ny, f.grid.Hy), :)

north_halo(f::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    view(f.data, :, right_halo_indices(LY, topology(f, 2), f.grid.Ny, f.grid.Hy), :)

bottom_halo(f::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    view(f.data, :, :, left_halo_indices(LZ, topology(f, 3), f.grid.Nz, f.grid.Hz), :)

bottom_halo(f::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    view(f.data, :, :, right_halo_indices(LZ, topology(f, 3), f.grid.Nz, f.grid.Hz), :)

underlying_west_halo(f, grid, location) =
    view(f.parent, underlying_left_halo_indices(location, topology(grid, 1), grid.Nx, grid.Hx), :, :)

underlying_east_halo(f, grid, location) =
    view(f.parent, underlying_right_halo_indices(location, topology(grid, 1), grid.Nx, grid.Hx), :, :)

underlying_south_halo(f, grid, location) =
    view(f.parent, :, underlying_left_halo_indices(location, topology(grid, 2), grid.Ny, grid.Hy), :)

underlying_north_halo(f, grid, location) =
    view(f.parent, :, underlying_right_halo_indices(location, topology(grid, 2), grid.Nz, grid.Hz), :)

underlying_bottom_halo(f, grid, location) =
    view(f.parent, :, :, underlying_left_halo_indices(location, topology(grid, 3), grid.Nz, grid.Hz))

underlying_top_halo(f, grid, location) =
    view(f.parent, :, :, underlying_right_halo_indices(location, topology(grid, 3), grid.Nz, grid.Hz))
