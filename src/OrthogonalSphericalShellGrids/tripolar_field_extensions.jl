using Oceananigans.BoundaryConditions: FieldBoundaryConditions,
                                       DefaultBoundaryCondition,
                                       regularize_boundary_condition,
                                       assumed_field_location,
                                       regularize_immersed_boundary_condition,
                                       LeftBoundary,
                                       RightBoundary
using Oceananigans.Grids: Grids, Center, Face,
                          LeftConnectedRightCenterFolded, LeftConnectedRightFaceFolded,
                          LeftConnectedRightCenterConnected, LeftConnectedRightFaceConnected,
                          SerialFoldedTopology
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

# Short alias for serial-tripolar dispatch signatures. Distributed counterparts
# (`SlabFTG`, `PencilFTG`, `DistFTG`) live in `distributed_tripolar_grid.jl`.
const SerialFTG = TripolarGridOfSomeKind{<:Any, <:Any, <:SerialFoldedTopology}

#####
##### North fold BC Regularization
#####

# We add `regularize_boundary_condition` methods that pass the zipper BC sign as an extra arg.

# DefaultBC on a serial tripolar → local `Zipper` with sign
BoundaryConditions.regularize_boundary_condition(::DefaultBoundaryCondition, grid::SerialFTG, loc, dim, bound, prognostic_names, sign) =
    north_fold_boundary_condition(grid)(sign)

# User-supplied BC on a serial tripolar → pass through (Field validates later).
# `bc::BoundaryCondition` disambiguates against the generic method at BoundaryConditions.jl:244.
BoundaryConditions.regularize_boundary_condition(bc::BoundaryCondition, grid::SerialFTG, loc, dim, bound, prognostic_names, sign) = bc


function BoundaryConditions.regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                                                 grid::TripolarGridOfSomeKind,
                                                                 field_name::Symbol,
                                                                 prognostic_names=nothing)

    loc  = assumed_field_location(field_name)
    sign = field_name == :u || field_name == :v ? -1 : 1

    west   = regularize_boundary_condition(bcs.west,   grid, loc, 1, LeftBoundary,  prognostic_names)
    east   = regularize_boundary_condition(bcs.east,   grid, loc, 1, RightBoundary, prognostic_names)
    south  = regularize_boundary_condition(bcs.south,  grid, loc, 2, LeftBoundary,  prognostic_names)
    north  = regularize_boundary_condition(bcs.north,  grid, loc, 2, RightBoundary, prognostic_names, sign)
    bottom = regularize_boundary_condition(bcs.bottom, grid, loc, 3, LeftBoundary,  prognostic_names)
    top    = regularize_boundary_condition(bcs.top,    grid, loc, 3, RightBoundary, prognostic_names)

    immersed = regularize_immersed_boundary_condition(bcs.immersed, grid, loc, field_name, prognostic_names)

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

BoundaryConditions.default_auxiliary_bc(grid::TripolarGridOfSomeKind, ::Val{:north}, loc) = north_fold_boundary_condition(grid)(1)
BoundaryConditions.default_auxiliary_bc(grid::TripolarGridOfSomeKind, ::Val{:north}, loc::Tuple{<:Any, Nothing, <:Any}) = nothing
