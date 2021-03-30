import Oceananigans.BoundaryConditions:
    fill_halo_regions!, fill_west_halo!, fill_east_halo!, fill_south_halo!, fill_north_halo!

# These filling functions won't work so let's not use them.

 fill_west_halo!(c, bc::CubedSphereExchangeBC, args...; kwargs...) = nothing
 fill_east_halo!(c, bc::CubedSphereExchangeBC, args...; kwargs...) = nothing
fill_south_halo!(c, bc::CubedSphereExchangeBC, args...; kwargs...) = nothing
fill_north_halo!(c, bc::CubedSphereExchangeBC, args...; kwargs...) = nothing

function fill_halo_regions!(field::ConformalCubedSphereField{LX, LY, LZ}, arch, args...) where {LX, LY, LZ}

    cubed_sphere_grid = field.grid

    for field_face in field.faces
        # Fill the top and bottom halos the usual way.
        fill_halo_regions!(field_face, arch, args...)

        # Deal with halo exchanges.
        fill_west_halo!(field_face, cubed_sphere_grid, field)
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

function cubed_sphere_boundary(cubed_sphere_field, location, face_number, side)
    src_field = cubed_sphere_field.faces[face_number]
    side == :west  && return  underlying_west_boundary(src_field.data, src_field.grid, location)
    side == :east  && return  underlying_east_boundary(src_field.data, src_field.grid, location)
    side == :south && return underlying_south_boundary(src_field.data, src_field.grid, location)
    side == :north && return underlying_north_boundary(src_field.data, src_field.grid, location)
end

function fill_west_halo!(field::ConformalCubedSphereFaceField{LX, LY, LZ}, cubed_sphere_grid::ConformalCubedSphereGrid, cubed_sphere_field) where {LX, LY, LZ}
    location = (LX, LY, LZ)
    dest_halo = underlying_west_halo(field.data, field.grid, location)

    exchange_info = field.boundary_conditions.west.condition
    src_face_number = exchange_info.to_face
    src_side = exchange_info.to_side
    src_boundary = cubed_sphere_boundary(cubed_sphere_field, location, src_face_number, src_side)

    if sides_in_the_same_dimension(:west, src_side)
        dest_halo .= src_boundary
    else
        dest_halo .= permutedims(src_boundary, (2, 1, 3))
    end

    return nothing
end

# # We're going to need this one for velocities...

# function fill_halo_regions!(fields::Union{Tuple, NamedTuple}, arch, args...)
#     for field in fields
#         fill_halo_regions!(field, arch, args...)
#     end
#     return nothing
# end
