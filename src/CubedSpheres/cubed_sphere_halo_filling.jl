import Oceananigans.BoundaryConditions:
    fill_halo_regions!, fill_west_halo!, fill_east_halo!, fill_south_halo!, fill_north_halo!

import Oceananigans.Models.HydrostaticFreeSurfaceModels: fill_horizontal_velocity_halos!

# These filling functions won't work so let's not use them.

 fill_west_halo!(c, bc::CubedSphereExchangeBC, args...; kwargs...) = nothing
 fill_east_halo!(c, bc::CubedSphereExchangeBC, args...; kwargs...) = nothing
fill_south_halo!(c, bc::CubedSphereExchangeBC, args...; kwargs...) = nothing
fill_north_halo!(c, bc::CubedSphereExchangeBC, args...; kwargs...) = nothing

function fill_halo_regions!(field::ConformalCubedSphereField{LX, LY, LZ}, arch, args...) where {LX, LY, LZ}

    location = (LX, LY, LZ)
    if location == (Face, Center, Center) || location == (Center, Face, Center)
        # @warn "Not filling halos for cubed sphere field with location $location. Use fill_horizontal_velocity_halos! for now."
        return nothing
    end

    cubed_sphere_grid = field.grid

    for field_face in field.faces
        # Fill the top and bottom halos the usual way.
        # Disable for a bit because errors.
        fill_halo_regions!(field_face, arch, args...)

        # Deal with halo exchanges.
        fill_west_halo!(field_face, cubed_sphere_grid, field)
        fill_east_halo!(field_face, cubed_sphere_grid, field)
        fill_south_halo!(field_face, cubed_sphere_grid, field)
        fill_north_halo!(field_face, cubed_sphere_grid, field)
    end

    return nothing
end

function sides_in_the_same_dimension(side1, side2)
    x_sides = (:west, :east)
    y_sides = (:south, :north)
    z_sides = (:bottom, :top)
    side1 in x_sides && side2 in x_sides && return true
    side1 in y_sides && side2 in y_sides && return true
    side1 in z_sides && side2 in z_sides && return true
    return false
end

function cubed_sphere_halo(cubed_sphere_field, location, face_number, side)
    LX, LY, LZ = location
    src_field = cubed_sphere_field.faces[face_number]
    side == :west  && return  underlying_west_halo(src_field.data, src_field.grid, LX)
    side == :east  && return  underlying_east_halo(src_field.data, src_field.grid, LX)
    side == :south && return underlying_south_halo(src_field.data, src_field.grid, LY)
    side == :north && return underlying_north_halo(src_field.data, src_field.grid, LY)
end

function cubed_sphere_boundary(cubed_sphere_field, location, face_number, side)
    LX, LY, LZ = location
    src_field = cubed_sphere_field.faces[face_number]
    side == :west  && return  underlying_west_boundary(src_field.data, src_field.grid, LX)
    side == :east  && return  underlying_east_boundary(src_field.data, src_field.grid, LX)
    side == :south && return underlying_south_boundary(src_field.data, src_field.grid, LY)
    side == :north && return underlying_north_boundary(src_field.data, src_field.grid, LY)
end

function fill_west_halo!(field::ConformalCubedSphereFaceField{LX, LY, LZ}, cubed_sphere_grid::ConformalCubedSphereGrid, cubed_sphere_field) where {LX, LY, LZ}
    location = (LX, LY, LZ)
    dest_halo = underlying_west_halo(field.data, field.grid, LX)

    exchange_info = field.boundary_conditions.west.condition
    src_face_number = exchange_info.to_face
    src_side = exchange_info.to_side
    src_boundary = cubed_sphere_boundary(cubed_sphere_field, location, src_face_number, src_side)

    if sides_in_the_same_dimension(:west, src_side)
        dest_halo .= src_boundary
    else
        dest_halo .= reverse(permutedims(src_boundary, (2, 1, 3)), dims=2)
    end

    return nothing
end

function fill_east_halo!(field::ConformalCubedSphereFaceField{LX, LY, LZ}, cubed_sphere_grid::ConformalCubedSphereGrid, cubed_sphere_field) where {LX, LY, LZ}
    location = (LX, LY, LZ)
    dest_halo = underlying_east_halo(field.data, field.grid, LX)

    exchange_info = field.boundary_conditions.east.condition
    src_face_number = exchange_info.to_face
    src_side = exchange_info.to_side
    src_boundary = cubed_sphere_boundary(cubed_sphere_field, location, src_face_number, src_side)

    if sides_in_the_same_dimension(:east, src_side)
        dest_halo .= src_boundary
    else
        dest_halo .= reverse(permutedims(src_boundary, (2, 1, 3)), dims=2)
    end

    return nothing
end

function fill_south_halo!(field::ConformalCubedSphereFaceField{LX, LY, LZ}, cubed_sphere_grid::ConformalCubedSphereGrid, cubed_sphere_field) where {LX, LY, LZ}
    location = (LX, LY, LZ)
    dest_halo = underlying_south_halo(field.data, field.grid, LY)

    exchange_info = field.boundary_conditions.south.condition
    src_face_number = exchange_info.to_face
    src_side = exchange_info.to_side
    src_boundary = cubed_sphere_boundary(cubed_sphere_field, location, src_face_number, src_side)

    if sides_in_the_same_dimension(:south, src_side)
        dest_halo .= src_boundary
    else
        dest_halo .= reverse(permutedims(src_boundary, (2, 1, 3)), dims=1)
    end

    return nothing
end

function fill_north_halo!(field::ConformalCubedSphereFaceField{LX, LY, LZ}, cubed_sphere_grid::ConformalCubedSphereGrid, cubed_sphere_field) where {LX, LY, LZ}
    location = (LX, LY, LZ)
    dest_halo = underlying_north_halo(field.data, field.grid, LY)

    exchange_info = field.boundary_conditions.north.condition
    src_face_number = exchange_info.to_face
    src_side = exchange_info.to_side
    src_boundary = cubed_sphere_boundary(cubed_sphere_field, location, src_face_number, src_side)

    if sides_in_the_same_dimension(:north, src_side)
        dest_halo .= src_boundary
    else
        dest_halo .= reverse(permutedims(src_boundary, (2, 1, 3)), dims=1)
    end

    return nothing
end

function fill_horizontal_velocity_halos!(u, v, arch)

    ## Fill them like they're tracers (mostly to get the top and bottom filled).
    ## Right now this errors because u_loc = fcc means that west/east halo and
    ## south/north halo sizes do not match.
    # fill_halo_regions!(u, arch)
    # fill_halo_regions!(v, arch)

    ## Now fill in the ones that need to be rotated.

    u_loc = (Face, Center, Center)
    v_loc = (Center, Face, Center)

    for face_number in 1:6, side in (:west, :east, :south, :north)
        exchange_info = getproperty(u.faces[face_number].boundary_conditions, side).condition
        src_face_number = exchange_info.to_face
        src_side = exchange_info.to_side

        if sides_in_the_same_dimension(side, src_side)
            cubed_sphere_halo(u, u_loc, face_number, side) .= cubed_sphere_boundary(u, u_loc, src_face_number, src_side)
            cubed_sphere_halo(v, v_loc, face_number, side) .= cubed_sphere_boundary(v, v_loc, src_face_number, src_side)
        else
            u_sign = (isodd(face_number) && side == :west ) || (iseven(face_number) && side == :east ) ? +1 : -1
            v_sign = (isodd(face_number) && side == :north) || (iseven(face_number) && side == :south) ? +1 : -1

            reverse_dim = src_side in (:west, :east) ? 1 : 2

            cubed_sphere_halo(u, u_loc, face_number, side) .= u_sign * reverse(permutedims(cubed_sphere_boundary(v, v_loc, src_face_number, src_side), (2, 1, 3)), dims=reverse_dim)
            cubed_sphere_halo(v, v_loc, face_number, side) .= v_sign * reverse(permutedims(cubed_sphere_boundary(u, u_loc, src_face_number, src_side), (2, 1, 3)), dims=reverse_dim)
        end
    end

    return nothing
end
