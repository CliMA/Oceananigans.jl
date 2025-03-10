using Oceananigans.Grids: architecture, cpu_face_constructor_z

import Oceananigans.Grids: with_halo

function with_halo(new_halo, old_grid::TripolarGrid)

    size = (old_grid.Nx, old_grid.Ny, old_grid.Nz)

    z = cpu_face_constructor_z(old_grid)

    north_poles_latitude = old_grid.conformal_mapping.north_poles_latitude
    first_pole_longitude = old_grid.conformal_mapping.first_pole_longitude
    southernmost_latitude = old_grid.conformal_mapping.southernmost_latitude

    new_grid = TripolarGrid(architecture(old_grid), eltype(old_grid);
                            size, z, halo = new_halo,
                            radius = old_grid.radius,
                            north_poles_latitude,
                            first_pole_longitude,
                            southernmost_latitude)

    return new_grid
end

function with_halo(new_halo, old_grid::DistributedTripolarGrid) 

    arch = old_grid.architecture

    n  = size(old_grid)
    N  = map(sum, concatenate_local_sizes(n, arch))
    z  = cpu_face_constructor_z(old_grid)

    north_poles_latitude = old_grid.conformal_mapping.north_poles_latitude
    first_pole_longitude = old_grid.conformal_mapping.first_pole_longitude
    southernmost_latitude = old_grid.conformal_mapping.southernmost_latitude

    return TripolarGrid(arch, eltype(old_grid);
                        halo = new_halo, 
                        size = N, 
                        north_poles_latitude,
                        first_pole_longitude,
                        southernmost_latitude,
                        z)
end
