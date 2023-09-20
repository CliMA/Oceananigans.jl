using Oceananigans.BoundaryConditions: DistributedCommunicationBoundaryCondition, FieldBoundaryConditions
using Oceananigans.BoundaryConditions: AbstractBoundaryConditionClassification
import Oceananigans.BoundaryConditions: bc_str

struct HaloCommunicationRanks{F, T}
    from :: F
      to :: T
end

HaloCommunicationRanks(; from, to) = HaloCommunicationRanks(from, to)

Base.summary(hcr::HaloCommunicationRanks) = "(from rank $(hcr.from) to rank $(hcr.to))"

function inject_halo_communication_boundary_conditions(field_bcs, local_rank, connectivity)
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

    west     = isnothing(rank_west)   ? field_bcs.west   : west_comm_bc
    east     = isnothing(rank_east)   ? field_bcs.east   : east_comm_bc
    south    = isnothing(rank_south)  ? field_bcs.south  : south_comm_bc
    north    = isnothing(rank_north)  ? field_bcs.north  : north_comm_bc
    
    bottom   = field_bcs.bottom 
    top      = field_bcs.top    
    immersed = field_bcs.immersed

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end
