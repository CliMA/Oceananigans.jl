using Oceananigans.BoundaryConditions: FieldBoundaryConditions,
                                       regularize_boundary_condition,
                                       assumed_field_location,
                                       regularize_immersed_boundary_condition,
                                       validate_boundary_condition_topology,
                                       LeftBoundary,
                                       RightBoundary
using Oceananigans.Grids: Grids, Center, Face,
                          LeftConnectedRightCenterFolded, LeftConnectedRightFaceFolded,
                          LeftConnectedRightCenterConnected, LeftConnectedRightFaceConnected
using Oceananigans.BoundaryConditions: BoundaryConditions

# A tripolar grid is always between 0 and 360 in longitude
# and always caps at the north pole (90°N)
Grids.x_domain(grid::TripolarGridOfSomeKind) = 0, 360
Grids.y_domain(grid::TripolarGridOfSomeKind) = minimum(parent(grid.φᶠᶠᵃ)), 90

# Determine the appropriate north fold boundary condition based on grid topology.
# Non-fold topologies (FullyConnected, RightConnected, etc.) default to UPivot — this
# value is only used as a placeholder; these ranks get their north BC overridden by
# inject_halo_communication_boundary_conditions or regularize_field_boundary_conditions.
north_fold_boundary_condition(::Type{<:AbstractTopology})                = UPivotZipperBoundaryCondition
north_fold_boundary_condition(::Type{RightCenterFolded})                 = UPivotZipperBoundaryCondition
north_fold_boundary_condition(::Type{LeftConnectedRightCenterFolded})    = UPivotZipperBoundaryCondition
north_fold_boundary_condition(::Type{LeftConnectedRightCenterConnected}) = UPivotZipperBoundaryCondition
north_fold_boundary_condition(::Type{RightFaceFolded})                   = FPivotZipperBoundaryCondition
north_fold_boundary_condition(::Type{LeftConnectedRightFaceFolded})      = FPivotZipperBoundaryCondition
north_fold_boundary_condition(::Type{LeftConnectedRightFaceConnected})   = FPivotZipperBoundaryCondition
north_fold_boundary_condition(grid::TripolarGridOfSomeKind) = north_fold_boundary_condition(topology(grid, 2))

# a `TripolarGrid` needs a `UPivotZipperBoundaryCondition` for the north boundary
# The `sign` 1 for regular tracers and -1 for velocities and signed vectors
"""
    regularize_field_boundary_conditions(bcs, grid::TripolarGridOfSomeKind, field_name, prognostic_names=nothing)

Regularize `bcs` for a tripolar grid, overriding `bcs.north` with the appropriate
`Zipper` boundary condition for the grid's y-topology. The north BC is topologically
required to be a `Zipper`: supplying any other BC (other than a distributed
communication BC or `nothing`) throws an `ArgumentError`. By default, fields named
`:u` or `:v` are treated as horizontal signed vectors (sign = -1); all other fields
use sign = +1.
"""
function BoundaryConditions.regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                                                 grid::TripolarGridOfSomeKind,
                                                                 field_name::Symbol,
                                                                 prognostic_names=nothing)

    validate_boundary_condition_topology(bcs.north, topology(grid, 2)(), :north)

    loc = assumed_field_location(field_name)

    west   = regularize_boundary_condition(bcs.west,   grid, loc, 1, LeftBoundary,  prognostic_names)
    east   = regularize_boundary_condition(bcs.east,   grid, loc, 1, RightBoundary, prognostic_names)
    south  = regularize_boundary_condition(bcs.south,  grid, loc, 2, LeftBoundary,  prognostic_names)

    # Assumption: :u and :v are signed vectors, all other fields are scalars
    sign = field_name == :u || field_name == :v ? -1 : 1
    north  = north_fold_boundary_condition(grid)(sign)

    bottom = regularize_boundary_condition(bcs.bottom, grid, loc, 3, LeftBoundary,  prognostic_names)
    top    = regularize_boundary_condition(bcs.top,    grid, loc, 3, RightBoundary, prognostic_names)

    immersed = regularize_immersed_boundary_condition(bcs.immersed, grid, loc, field_name, prognostic_names)

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

BoundaryConditions.default_auxiliary_bc(grid::TripolarGridOfSomeKind, ::Val{:north}, loc) = north_fold_boundary_condition(grid)(1)
BoundaryConditions.default_auxiliary_bc(grid::TripolarGridOfSomeKind, ::Val{:north}, loc::Tuple{<:Any, Nothing, <:Any}) = nothing
