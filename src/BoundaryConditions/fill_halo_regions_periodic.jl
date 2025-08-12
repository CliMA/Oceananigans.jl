using KernelAbstractions.Extras.LoopInfo: @unroll

#####
##### Periodic boundary conditions
#####

function fill_west_and_east_halo!(c, ::PBCT, ::PBCT, size, offset, loc, arch, grid, args...; only_local_halos = false, kw...)
    c_parent, yz_size, offset = periodic_size_and_offset(c, 2, 3, size, offset)
    launch!(arch, grid, KernelParameters(yz_size, offset), _fill_periodic_west_and_east_halo!, c, Val(grid.Hx), grid.Nx; kw...)
    return nothing
end

function fill_south_and_north_halo!(c, ::PBCT, ::PBCT, size, offset, loc, arch, grid, args...; only_local_halos = false, kw...)
    c_parent, xz_size, offset = periodic_size_and_offset(c, 1, 3, size, offset)
    launch!(arch, grid, KernelParameters(xz_size, offset), _fill_periodic_south_and_north_halo!, c, Val(grid.Hy), grid.Ny;  kw...)
    return nothing
end

function fill_bottom_and_top_halo!(c, ::PBCT, ::PBCT, size, offset, loc, arch, grid, args...; only_local_halos = false, kw...)
    c_parent, xy_size, offset = periodic_size_and_offset(c, 1, 2, size, offset)
    launch!(arch, grid, KernelParameters(xy_size, offset), _fill_periodic_bottom_and_top_halo!, c, Val(grid.Hz), grid.Nz; kw...)
    return nothing
end

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