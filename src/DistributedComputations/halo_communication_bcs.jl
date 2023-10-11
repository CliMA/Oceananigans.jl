using Oceananigans.BoundaryConditions: DistributedCommunicationBoundaryCondition, FieldBoundaryConditions
using Oceananigans.BoundaryConditions: AbstractBoundaryConditionClassification
import Oceananigans.BoundaryConditions: bc_str

struct HaloCommunicationRanks{F, T}
    from :: F
      to :: T
end

HaloCommunicationRanks(; from, to) = HaloCommunicationRanks(from, to)

Base.summary(hcr::HaloCommunicationRanks) = "(from rank $(hcr.from) to rank $(hcr.to))"

function inject_halo_communication_boundary_conditions(field_bcs, local_rank, connectivity, topology)
    rank_east   = connectivity.east
    rank_west   = connectivity.west
    rank_north  = connectivity.north
    rank_south  = connectivity.south

    east_comm_ranks   = HaloCommunicationRanks(from=local_rank, to=rank_east)
    west_comm_ranks   = HaloCommunicationRanks(from=local_rank, to=rank_west)
    north_comm_ranks  = HaloCommunicationRanks(from=local_rank, to=rank_north)
    south_comm_ranks  = HaloCommunicationRanks(from=local_rank, to=rank_south)

    east_comm_bc   = DistributedCommunicationBoundaryCondition(east_comm_ranks)
    west_comm_bc   = DistributedCommunicationBoundaryCondition(west_comm_ranks)
    north_comm_bc  = DistributedCommunicationBoundaryCondition(north_comm_ranks)
    south_comm_bc  = DistributedCommunicationBoundaryCondition(south_comm_ranks)

    TX, TY, _ = topology

    # `rank == nothing`indicates no partitioning in that specific direction.
    # Communication is required only if the direction is "connected" 
    # Remember `RightConnected` means bounded on the left and viceversa
    # `LeftConnected` means bounded on the right
    inject_west  = !isnothing(rank_west)  && !(TX isa RightConnected) 
    inject_east  = !isnothing(rank_east)  && !(TX isa LeftConnected) 
    inject_south = !isnothing(rank_south) && !(TY isa RightConnected) 
    inject_north = !isnothing(rank_north) && !(TY isa LeftConnected) 

    west     = inject_west  ? west_comm_bc  : field_bcs.west  
    east     = inject_east  ? east_comm_bc  : field_bcs.east  
    south    = inject_south ? south_comm_bc : field_bcs.south 
    north    = inject_north ? north_comm_bc : field_bcs.north 
    
    bottom   = field_bcs.bottom 
    top      = field_bcs.top    
    immersed = field_bcs.immersed

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end
