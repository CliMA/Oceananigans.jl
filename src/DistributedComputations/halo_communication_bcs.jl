using Oceananigans.BoundaryConditions:
    DistributedCommunicationBoundaryCondition,
    FieldBoundaryConditions

function inject_halo_communication_boundary_conditions(field_bcs, loc, local_rank, layout)
    communication = DistributedCommunicationBoundaryCondition(nothing)

    west  = communicates(layout, local_rank, West(),  loc) ? communication : field_bcs.west
    east  = communicates(layout, local_rank, East(),  loc) ? communication : field_bcs.east
    south = communicates(layout, local_rank, South(), loc) ? communication : field_bcs.south
    north = communicates(layout, local_rank, North(), loc) ? communication : field_bcs.north

    return FieldBoundaryConditions(west, east, south, north, field_bcs.bottom, field_bcs.top, field_bcs.immersed)
end
