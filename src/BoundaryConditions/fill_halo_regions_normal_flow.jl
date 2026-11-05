# Halo faces that stencils read beyond the boundary: one on an Open side, none on a Closed side
@inline left_halo_faces(::Type{<:TwoSided{<:Open}}) = 1
@inline left_halo_faces(topology) = 0

@inline right_halo_faces(::Type{<:TwoSided{<:Any, <:Open}}) = 1
@inline right_halo_faces(topology) = 0

# Normal-flow boundary fill: set the boundary-face value, and extend it to the halo faces for Open topologies
@inline   _fill_west_halo!(j, k, grid, c, bc::NFBC, loc, args...) = @inbounds c[1 - left_halo_faces(topology(grid, 1)), j, k]            = c[1, j, k]           = getbc(bc, j, k, grid, args...)
@inline   _fill_east_halo!(j, k, grid, c, bc::NFBC, loc, args...) = @inbounds c[grid.Nx + 1 + right_halo_faces(topology(grid, 1)), j, k] = c[grid.Nx + 1, j, k] = getbc(bc, j, k, grid, args...)
@inline  _fill_south_halo!(i, k, grid, c, bc::NFBC, loc, args...) = @inbounds c[i, 1 - left_halo_faces(topology(grid, 2)), k]            = c[i, 1, k]           = getbc(bc, i, k, grid, args...)
@inline  _fill_north_halo!(i, k, grid, c, bc::NFBC, loc, args...) = @inbounds c[i, grid.Ny + 1 + right_halo_faces(topology(grid, 2)), k] = c[i, grid.Ny + 1, k] = getbc(bc, i, k, grid, args...)
@inline _fill_bottom_halo!(i, j, grid, c, bc::NFBC, loc, args...) = @inbounds c[i, j, 1 - left_halo_faces(topology(grid, 3))]            = c[i, j, 1]           = getbc(bc, i, j, grid, args...)
@inline    _fill_top_halo!(i, j, grid, c, bc::NFBC, loc, args...) = @inbounds c[i, j, grid.Nz + 1 + right_halo_faces(topology(grid, 3))] = c[i, j, grid.Nz + 1] = getbc(bc, i, j, grid, args...)

# Normal-flow halos are skipped when `fill_normal_flow_bcs=false`
@inline fills_halo(::NFBC, fill_normal_flow_bcs) = fill_normal_flow_bcs
