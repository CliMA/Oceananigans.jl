using Oceananigans.Grids: left_halo_indices, right_halo_indices
using Oceananigans.Fields: AbstractField

@inline west_halo(f::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    view(f.data, left_halo_indices(LX, topology(f, 1), f.grid.Nx, f.grid.Hx), :, :)

@inline east_halo(f::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    view(f.data, right_halo_indices(LX, topology(f, 1), f.grid.Nx, f.grid.Hx), :, :)

@inline south_halo(f::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    view(f.data, :, left_halo_indices(LY, topology(f, 2), f.grid.Ny, f.grid.Hy), :)

@inline north_halo(f::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    view(f.data, :, right_halo_indices(LY, topology(f, 2), f.grid.Ny, f.grid.Hy), :)

@inline bottom_halo(f::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    view(f.data, :, :, left_halo_indices(LZ, topology(f, 3), f.grid.Nz, f.grid.Hz), :)

@inline bottom_halo(f::AbstractField{LX, LY, LZ}) where {LX, LY, LZ} =
    view(f.data, :, :, right_halo_indices(LZ, topology(f, 3), f.grid.Nz, f.grid.Hz), :)
