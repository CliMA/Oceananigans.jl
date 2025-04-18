using Oceananigans.BoundaryConditions: FieldBoundaryConditions, 
                                       assumed_field_location, 
                                       regularize_boundary_condition,
                                       regularize_immersed_boundary_condition,
                                       LeftBoundary,
                                       RightBoundary,
                                       BoundaryCondition,
                                       ZipperBoundaryCondition,
                                       Zipper,
                                       ZBC

using Oceananigans.Fields: architecture, 
                           validate_indices, 
                           validate_boundary_conditions,
                           validate_field_data, 
                           CommunicationBuffers

using Oceananigans.ImmersedBoundaries

import Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
import Oceananigans.Grids: x_domain, y_domain
import Oceananigans.Fields: Field
import Oceananigans.Fields: tupled_fill_halo_regions!

# A tripolar grid is always between 0 and 360 in longitude
# and always caps at the north pole (90°N)
x_domain(grid::TripolarGridOfSomeKind) = 0, 360
y_domain(grid::TripolarGridOfSomeKind) = minimum(parent(grid.φᶠᶠᵃ)), 90

# a `TripolarGrid` needs a `ZipperBoundaryCondition` for the north boundary
# The `sign` 1 for regular tracers and -1 for velocities and signed vectors
function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                              grid::TripolarGridOfSomeKind,
                                              field_name::Symbol,
                                              prognostic_names=nothing)

    loc = assumed_field_location(field_name)

    # Assumption: :u and :v are signed vectors, all other fields are scalars
    sign = field_name == :u || field_name == :v ? -1 : 1

    west   = regularize_boundary_condition(bcs.west,   grid, loc, 1, LeftBoundary,  prognostic_names)
    east   = regularize_boundary_condition(bcs.east,   grid, loc, 1, RightBoundary, prognostic_names)
    south  = regularize_boundary_condition(bcs.south,  grid, loc, 2, LeftBoundary,  prognostic_names)
    bottom = regularize_boundary_condition(bcs.bottom, grid, loc, 3, LeftBoundary,  prognostic_names)
    top    = regularize_boundary_condition(bcs.top,    grid, loc, 3, RightBoundary, prognostic_names)

    north  = ZipperBoundaryCondition(sign)

    immersed = regularize_immersed_boundary_condition(bcs.immersed, grid, loc, field_name, prognostic_names)

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

# HEAVY ASSUMPTION!!!!
# Fields living on edges are signed vectors while fields living on nodes and centers are scalars
sign(LX, LY) = 1
sign(::Type{Face},   ::Type{Face})   = 1
sign(::Type{Face},   ::Type{Center}) = - 1 # u-velocity type
sign(::Type{Center}, ::Type{Face})   = - 1 # v-velocity type
sign(::Type{Center}, ::Type{Center}) = 1

# Extension of the constructor for a `Field` on a `TripolarGridOfSomeKind` grid. We assumes that the north boundary is a zipper
# with a sign that depends on the location of the field (revert the value of the halos if on edges, keep it if on nodes or centers)
function Field((LX, LY, LZ)::Tuple, grid::TripolarGridOfSomeKind, data, old_bcs, indices::Tuple, op, status)
    indices = validate_indices(indices, (LX, LY, LZ), grid)
    validate_field_data((LX, LY, LZ), data, grid, indices)
    validate_boundary_conditions((LX, LY, LZ), grid, old_bcs)

    if isnothing(old_bcs) || ismissing(old_bcs)
        new_bcs = old_bcs
    else
        default_zipper = ZipperBoundaryCondition(sign(LX, LY))

        north_bc = old_bcs.north isa BoundaryCondition{<:Zipper} ? old_bcs.north : default_zipper

        new_bcs = FieldBoundaryConditions(; west = old_bcs.west, 
                                            east = old_bcs.east, 
                                            south = old_bcs.south,
                                            north = north_bc,
                                            top = old_bcs.top,
                                            bottom = old_bcs.bottom)
    end

    buffers = CommunicationBuffers(grid, data, new_bcs)

    return Field{LX, LY, LZ}(grid, data, new_bcs, indices, op, status, buffers)
end

# Not sure this is needed, but it is here for now
function tupled_fill_halo_regions!(full_fields, grid::TripolarGridOfSomeKind, args...; kwargs...)
    for field in full_fields
        fill_halo_regions!(field, args...; kwargs...)
    end
end