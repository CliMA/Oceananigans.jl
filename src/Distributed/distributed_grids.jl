using Oceananigans.Grids
using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, pop_flat_elements


@inline get_local_coords(c::Tuple         , nc, lc, index) = (c[1] + (index-1) * lc,    c[1] + index * lc)
@inline get_local_coords(c::AbstractVector, nc, lc, index) = c[1 + (index-1) * nc : 1 + nc * index]

@inline my_grid(grid::RectilinearGrid, x, y, z, size, halo) = RectilinearGrid(size = size, 
                                                                  halo = halo,
                                                          architecture = grid.architecture,
                                                              topology = topology(grid),
                                                                     x = x,
                                                                     y = y,
                                                                     z = z) 

@inline my_grid(grid::LatitudeLongitudeGrid, x, y, z, size, halo) = LatitudeLongitudeGrid(size = size, 
                                                                     halo = halo,
                                                             architecture = grid.architecture,
                                                                longitude = x,
                                                                 latitude = y,
                                                                        z = z) 
   

#### Returning the local Grids
function local_grids(local_index, ranks, connectivity, grid)
    i, j, k    = local_index
    Rx, Ry, Rz = ranks

    Nx, Ny, Nz = size(grid)
    Lx, Ly, Lz = length(grid)
    
    # Pull out face grid constructors
    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    # Make sure we can put an integer number of grid points in each rank.
    # Will generalize in the future.
    @assert isinteger(Nx / Rx)
    @assert isinteger(Ny / Ry)
    @assert isinteger(Nz / Rz)
    
    nx, ny, nz = Nx÷Rx, Ny÷Ry, Nz÷Rz
    lx, ly, lz = Lx/Rx, Ly/Ry, Lz/Rz

    xl = get_local_coords(x, nx, lx, i)
    yl = get_local_coords(y, ny, ly, j)
    zl = get_local_coords(z, nz, lz, k)

    topo       = topology(grid)
    local_size = pop_flat_elements((nx, ny, nz), topo)
    local_halo = pop_flat_elements(halo_size(grid), topo)

    # FIXME: local grid might have different topology!
    local_grid = my_grid(grid, xl, yl, zl, local_size, local_halo)

    return local_grid
end

 