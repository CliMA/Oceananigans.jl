using Oceananigans.BoundaryConditions: FieldBoundaryConditions, 
                                       regularize_boundary_condition,
                                       assumed_field_location,
                                       regularize_immersed_boundary_condition,
                                       LeftBoundary,
                                       RightBoundary

using Oceananigans.ImmersedBoundaries

import Oceananigans.BoundaryConditions: default_auxiliary_bc, regularize_field_boundary_conditions
import Oceananigans.Grids: x_domain, y_domain

# A tripolar grid is always between 0 and 360 in longitude
# and always caps at the north pole (90°N)
x_domain(grid::TripolarGridOfSomeKind) = 0, 360
y_domain(grid::TripolarGridOfSomeKind) = minimum(parent(grid.φᶠᶠᵃ)), 90

# Fields living on edges are signed vectors while fields living on nodes and centers are scalars
sign(LX, LY) = 1
sign(::Type{Face},   ::Type{Face})   = 1
sign(::Type{Face},   ::Type{Center}) = - 1 # u-velocity type
sign(::Type{Center}, ::Type{Face})   = - 1 # v-velocity type
sign(::Type{Center}, ::Type{Center}) = 1

# a `TripolarGrid` needs a `ZipperBoundaryCondition` for the north boundary
# The `sign` 1 for regular tracers and -1 for velocities and signed vectors
function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                              grid::TripolarGridOfSomeKind,
                                              field_name::Symbol,
                                              prognostic_names=nothing)

    loc = assumed_field_location(field_name)

    west   = regularize_boundary_condition(bcs.west,   grid, loc, 1, LeftBoundary,  prognostic_names)
    east   = regularize_boundary_condition(bcs.east,   grid, loc, 1, RightBoundary, prognostic_names)
    south  = regularize_boundary_condition(bcs.south,  grid, loc, 2, LeftBoundary,  prognostic_names)

    # Assumption: :u and :v are signed vectors, all other fields are scalars
    sign = field_name == :u || field_name == :v ? -1 : 1
    north  = ZipperBoundaryCondition(sign)

    bottom = regularize_boundary_condition(bcs.bottom, grid, loc, 3, LeftBoundary,  prognostic_names)
    top    = regularize_boundary_condition(bcs.top,    grid, loc, 3, RightBoundary, prognostic_names)

    immersed = regularize_immersed_boundary_condition(bcs.immersed, grid, loc, field_name, prognostic_names)

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

default_auxiliary_bc(grid::TripolarGridOfSomeKind, ::Val{:north}, loc) = ZipperBoundaryCondition(1)
