# The boundary value, also on the halo faces that belong to the domain
@inline   _fill_west_halo!(j, k, grid, c, bc::NFBC, loc, args...) = @inbounds c[1 - left_domain_halo(topology(grid, 1)), j, k]            = c[1, j, k]           = getbc(bc, j, k, grid, args...)
@inline   _fill_east_halo!(j, k, grid, c, bc::NFBC, loc, args...) = @inbounds c[grid.Nx + 1 + right_domain_halo(topology(grid, 1)), j, k] = c[grid.Nx + 1, j, k] = getbc(bc, j, k, grid, args...)
@inline  _fill_south_halo!(i, k, grid, c, bc::NFBC, loc, args...) = @inbounds c[i, 1 - left_domain_halo(topology(grid, 2)), k]            = c[i, 1, k]           = getbc(bc, i, k, grid, args...)
@inline  _fill_north_halo!(i, k, grid, c, bc::NFBC, loc, args...) = @inbounds c[i, grid.Ny + 1 + right_domain_halo(topology(grid, 2)), k] = c[i, grid.Ny + 1, k] = getbc(bc, i, k, grid, args...)
@inline _fill_bottom_halo!(i, j, grid, c, bc::NFBC, loc, args...) = @inbounds c[i, j, 1 - left_domain_halo(topology(grid, 3))]            = c[i, j, 1]           = getbc(bc, i, j, grid, args...)
@inline    _fill_top_halo!(i, j, grid, c, bc::NFBC, loc, args...) = @inbounds c[i, j, grid.Nz + 1 + right_domain_halo(topology(grid, 3))] = c[i, j, grid.Nz + 1] = getbc(bc, i, j, grid, args...)

# Normal-flow halos are skipped when `fill_normal_flow_bcs=false`
@inline fills_halo(::NFBC, fill_normal_flow_bcs) = fill_normal_flow_bcs
