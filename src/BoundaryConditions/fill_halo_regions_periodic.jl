#####
##### Periodic boundary condition kernels
#####

@kernel function _fill_periodic_west_and_east_halo!(c, west_bc, east_bc, loc, grid, args)
    j, k = @index(Global, NTuple)
    H = grid.Hx
    N = grid.Nx
    @inbounds for i = 1:H
        parent(c)[i, j, k]     = parent(c)[N+i, j, k] # west
        parent(c)[N+H+i, j, k] = parent(c)[H+i, j, k] # east
    end
end

@kernel function _fill_periodic_south_and_north_halo!(c, south_bc, north_bc, loc, grid, args) 
    i, k = @index(Global, NTuple)
    H = grid.Hy
    N = grid.Ny
    @inbounds for j = 1:H
        parent(c)[i, j, k]     = parent(c)[i, N+j, k] # south
        parent(c)[i, N+H+j, k] = parent(c)[i, H+j, k] # north
    end
end

@kernel function _fill_periodic_bottom_and_top_halo!(c, bottom_bc, top_bc, loc, grid, args)
    i, j = @index(Global, NTuple)
    H = grid.Hz
    N = grid.Nz
    @inbounds for k = 1:H
        parent(c)[i, j, k]     = parent(c)[i, j, N+k] # top
        parent(c)[i, j, N+H+k] = parent(c)[i, j, H+k] # bottom
    end
end