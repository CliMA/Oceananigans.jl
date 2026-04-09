using Oceananigans.BoundaryConditions:
    DistributedCommunicationBoundaryCondition,
    FieldBoundaryConditions,
    BoundaryCondition,
    MultiRegionCommunication,
    Flux

struct HaloCommunicationRanks{F, T}
    from :: F
      to :: T
end

HaloCommunicationRanks(; from, to) = HaloCommunicationRanks(from, to)

Base.summary(hcr::HaloCommunicationRanks) = "HaloCommunicationRanks from rank $(hcr.from) to rank $(hcr.to)"

# Zero-flux BC used for empty-tile neighbors in load-balanced layouts.
# When a tile is absent, the halo is inland (fully immersed), so zero-flux is safe.
const ZFBC_instance = BoundaryCondition(Flux(), nothing)

function inject_halo_communication_boundary_conditions(field_bcs, loc, local_rank, connectivity, topology)
    rank_east   = connectivity.east
    rank_west   = connectivity.west
    rank_north  = connectivity.north
    rank_south  = connectivity.south

    east_comm_ranks  = HaloCommunicationRanks(from=local_rank, to=rank_east)
    west_comm_ranks  = HaloCommunicationRanks(from=local_rank, to=rank_west)
    north_comm_ranks = HaloCommunicationRanks(from=local_rank, to=rank_north)
    south_comm_ranks = HaloCommunicationRanks(from=local_rank, to=rank_south)

    east_comm_bc  = DistributedCommunicationBoundaryCondition(east_comm_ranks)
    west_comm_bc  = DistributedCommunicationBoundaryCondition(west_comm_ranks)
    north_comm_bc = DistributedCommunicationBoundaryCondition(north_comm_ranks)
    south_comm_bc = DistributedCommunicationBoundaryCondition(south_comm_ranks)

    TX, TY, _ = topology

    # A direction is "connected" when the topology says it should have a neighbor
    # (i.e. it's not a global boundary). `RightConnected` = bounded on the left,
    # `LeftConnected` = bounded on the right.
    west_connected  = (TX != RightConnected) && !isnothing(loc[1])
    east_connected  = (TX != LeftConnected)  && !isnothing(loc[1])
    south_connected = (TY != RightConnected) && !isnothing(loc[2])
    north_connected = (TY != LeftConnected)  && !isnothing(loc[2])

    # For connected directions: use MPI communication if the neighbor exists,
    # or zero-flux if the neighbor is absent (empty tile in a load-balanced layout).
    # For non-connected directions (global boundaries): keep the field's original BC.
    west  = west_connected  ? (isnothing(rank_west)  ? ZFBC_instance : west_comm_bc)  : field_bcs.west
    east  = east_connected  ? (isnothing(rank_east)  ? ZFBC_instance : east_comm_bc)  : field_bcs.east
    south = south_connected ? (isnothing(rank_south) ? ZFBC_instance : south_comm_bc) : field_bcs.south
    north = north_connected ? (isnothing(rank_north) ? ZFBC_instance : north_comm_bc) : field_bcs.north

    bottom   = field_bcs.bottom
    top      = field_bcs.top
    immersed = field_bcs.immersed

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end
