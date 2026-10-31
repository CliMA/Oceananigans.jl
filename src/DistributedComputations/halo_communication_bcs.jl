using Oceananigans.BoundaryConditions:
    DistributedCommunicationBoundaryCondition,
    FieldBoundaryConditions,
    DCBC,
    ZBC

# The sign of the fold, held by the zipper and by the halo communication that replaces it
zipper_sign(bc::Union{ZBC, DCBC}) = bc.condition
zipper_sign(bc) = nothing

inject_halo_communication_boundary_conditions(::Nothing, loc, local_rank, connectivity) = nothing
inject_halo_communication_boundary_conditions(::Missing, loc, local_rank, connectivity) = missing

function inject_halo_communication_boundary_conditions(field_bcs, loc, local_rank, connectivity)
    communicating = communicating_sides(connectivity, local_rank, loc)
    communication = DistributedCommunicationBoundaryCondition(nothing)

    west  = communicating.west  ? communication : field_bcs.west
    east  = communicating.east  ? communication : field_bcs.east
    south = communicating.south ? communication : field_bcs.south
    north = communicating.north ? DistributedCommunicationBoundaryCondition(zipper_sign(field_bcs.north)) : field_bcs.north

    return FieldBoundaryConditions(west, east, south, north, field_bcs.bottom, field_bcs.top, field_bcs.immersed)
end
