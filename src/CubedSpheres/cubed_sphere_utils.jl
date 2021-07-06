using Oceananigans.Fields: AbstractField
using Oceananigans.Grids:
    Face, Bounded,
    interior_indices,
    left_halo_indices, right_halo_indices,
    underlying_left_halo_indices, underlying_right_halo_indices

# TODO: Move to Grids/grid_utils.jl

#####
##### Viewing halos
#####

west_halo(f::AbstractField{LX, LY, LZ}; include_corners=true) where {LX, LY, LZ} =
    include_corners ? view(f.data, left_halo_indices(LX, topology(f, 1), f.grid.Nx, f.grid.Hx), :, :) :
                      view(f.data, left_halo_indices(LX, topology(f, 1), f.grid.Nx, f.grid.Hx),
                                   interior_indices(LY, topology(f, 2), f.grid.Ny),
                                   interior_indices(LZ, topology(f, 3), f.grid.Nz))

east_halo(f::AbstractField{LX, LY, LZ}; include_corners=true) where {LX, LY, LZ} =
    include_corners ? view(f.data, right_halo_indices(LX, topology(f, 1), f.grid.Nx, f.grid.Hx), :, :) :
                      view(f.data, right_halo_indices(LX, topology(f, 1), f.grid.Nx, f.grid.Hx),
                                   interior_indices(LY, topology(f, 2), f.grid.Ny),
                                   interior_indices(LZ, topology(f, 3), f.grid.Nz))

south_halo(f::AbstractField{LX, LY, LZ}; include_corners=true) where {LX, LY, LZ} =
    include_corners ? view(f.data, :, left_halo_indices(LY, topology(f, 2), f.grid.Ny, f.grid.Hy), :) :
                      view(f.data, interior_indices(LX, topology(f, 1), f.grid.Nx),
                                   left_halo_indices(LY, topology(f, 2), f.grid.Ny, f.grid.Hy),
                                   interior_indices(LZ, topology(f, 3), f.grid.Nz))

north_halo(f::AbstractField{LX, LY, LZ}; include_corners=true) where {LX, LY, LZ} =
    include_corners ? view(f.data, :, right_halo_indices(LY, topology(f, 2), f.grid.Ny, f.grid.Hy), :) :
                      view(f.data, interior_indices(LX, topology(f, 1), f.grid.Nx),
                                   right_halo_indices(LY, topology(f, 2), f.grid.Ny, f.grid.Hy),
                                   interior_indices(LZ, topology(f, 3), f.grid.Nz))

bottom_halo(f::AbstractField{LX, LY, LZ}; include_corners=true) where {LX, LY, LZ} =
    include_corners ? view(f.data, :, :, left_halo_indices(LZ, topology(f, 3), f.grid.Nz, f.grid.Hz)) :
                      view(f.data, interior_indices(LX, topology(f, 1), f.grid.Nx),
                                   interior_indices(LY, topology(f, 2), f.grid.Ny),
                                   left_halo_indices(LZ, topology(f, 3), f.grid.Nz, f.grid.Hz))

top_halo(f::AbstractField{LX, LY, LZ}; include_corners=true) where {LX, LY, LZ} =
    include_corners ? view(f.data, :, :, right_halo_indices(LZ, topology(f, 3), f.grid.Nz, f.grid.Hz)) :
                      view(f.data, interior_indices(LX, topology(f, 1), f.grid.Nx),
                                   interior_indices(LY, topology(f, 2), f.grid.Ny),
                                   right_halo_indices(LZ, topology(f, 3), f.grid.Nz, f.grid.Hz))

underlying_west_halo(f, grid, location, topo=topology(grid, 1)) =
    view(f.parent, underlying_left_halo_indices(location, topo, grid.Nx, grid.Hx), :, :)

underlying_east_halo(f, grid, location, topo=topology(grid, 1)) =
    view(f.parent, underlying_right_halo_indices(location, topo, grid.Nx, grid.Hx), :, :)

underlying_south_halo(f, grid, location, topo=topology(grid, 2)) =
    view(f.parent, :, underlying_left_halo_indices(location, topo, grid.Ny, grid.Hy), :)

underlying_north_halo(f, grid, location, topo=topology(grid, 2)) =
    view(f.parent, :, underlying_right_halo_indices(location, topo, grid.Ny, grid.Hy), :)

underlying_bottom_halo(f, grid, location, topo=topology(grid, 3)) =
    view(f.parent, :, :, underlying_left_halo_indices(location, topo, grid.Nz, grid.Hz))

underlying_top_halo(f, grid, location, topo=topology(grid, 3)) =
    view(f.parent, :, :, underlying_right_halo_indices(location, topo, grid.Nz, grid.Hz))

#####
##### Viewing boundary grid points (used to fill other halos)
#####

left_boundary_indices(loc, topo, N, H) = 1:H
left_boundary_indices(::Type{Nothing}, topo, N, H) = 1:0 # empty

right_boundary_indices(loc, topo, N, H) = N-H+1:N
right_boundary_indices(::Type{Face}, ::Type{Bounded}, N, H) = N-H:N+1
right_boundary_indices(::Type{Nothing}, topo, N, H) = 1:0 # empty

underlying_left_boundary_indices(loc, topo, N, H) = 1+H:2H
underlying_left_boundary_indices(::Type{Nothing}, topo, N, H) = 1:0 # empty

underlying_right_boundary_indices(loc, topo, N, H) = N+1:N+H
underlying_right_boundary_indices(::Type{Face}, ::Type{Bounded}, N, H) = N+2:N+H+1
underlying_right_boundary_indices(::Type{Nothing}, topo, N, H) = 1:0 # empty

underlying_west_boundary(f, grid, location, topo=topology(grid, 1)) =
    view(f.parent, underlying_left_boundary_indices(location, topo, grid.Nx, grid.Hx), :, :)

underlying_east_boundary(f, grid, location, topo=topo = topology(grid, 1)) =
    view(f.parent, underlying_right_boundary_indices(location, topo, grid.Nx, grid.Hx), :, :)

underlying_south_boundary(f, grid, location, topo=topology(grid, 2)) =
    view(f.parent, :, underlying_left_boundary_indices(location, topo, grid.Ny, grid.Hy), :)

underlying_north_boundary(f, grid, location, topo=topology(grid, 2)) =
    view(f.parent, :, underlying_right_boundary_indices(location, topo, grid.Ny, grid.Hy), :)

underlying_bottom_boundary(f, grid, location, topo=topology(grid, 3)) =
    view(f.parent, :, :, underlying_left_boundary_indices(location, topo, grid.Nz, grid.Hz))

underlying_top_boundary(f, grid, location, topo=topology(grid, 3)) =
    view(f.parent, :, :, underlying_right_boundary_indices(location, topo, grid.Nz, grid.Hz))

#####
##### Convinience functions
#####

function sides_in_the_same_dimension(side1, side2)
    x_sides = (:west, :east)
    y_sides = (:south, :north)
    z_sides = (:bottom, :top)
    side1 in x_sides && side2 in x_sides && return true
    side1 in y_sides && side2 in y_sides && return true
    side1 in z_sides && side2 in z_sides && return true
    return false
end

function cubed_sphere_halo(cubed_sphere_field, location, face_index, side)
    LX, LY, LZ = location
    src_field = get_face(cubed_sphere_field, face_index)
    side == :west  && return  underlying_west_halo(src_field.data, src_field.grid, LX)
    side == :east  && return  underlying_east_halo(src_field.data, src_field.grid, LX)
    side == :south && return underlying_south_halo(src_field.data, src_field.grid, LY)
    side == :north && return underlying_north_halo(src_field.data, src_field.grid, LY)
end

function cubed_sphere_boundary(cubed_sphere_field, location, face_index, side)
    LX, LY, LZ = location
    src_field = get_face(cubed_sphere_field, face_index)
    side == :west  && return  underlying_west_boundary(src_field.data, src_field.grid, LX)
    side == :east  && return  underlying_east_boundary(src_field.data, src_field.grid, LX)
    side == :south && return underlying_south_boundary(src_field.data, src_field.grid, LY)
    side == :north && return underlying_north_boundary(src_field.data, src_field.grid, LY)
end

function shifted_fill!(dest, src; i_shift=0, j_shift=0)
    dest[1+i_shift:end, 1+j_shift:end, :] .= src[1:end-i_shift, 1:end-j_shift, :]
    return nothing
end
