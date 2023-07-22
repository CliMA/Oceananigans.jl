using Oceananigans.Grids: topology

"""
    struct RegionalConnectivity{S <: AbstractRegionSide, FS <: AbstractRegionSide} <: AbstractConnectivity

The connectivity among various regions in a multi-region partition.

$(TYPEDFIELDS)
"""
struct RegionalConnectivity{S <: AbstractRegionSide, FS <: AbstractRegionSide} <: AbstractConnectivity
    "the current region rank"
         rank :: Int
    "the region from which boundary condition comes from"
    from_rank :: Int
    "the current region side"
         side :: S
    "the side of the region from which boundary condition comes from"
    from_side :: FS
end

function Connectivity(devices, partition::Union{XPartition, YPartition}, global_grid::AbstractGrid)
    regions = MultiRegionObject(Tuple(1:length(devices)), devices)
    @apply_regionally connectivity = find_regional_connectivities(regions, partition, global_grid)
    return connectivity
end

function find_regional_connectivities(region, partition, global_grid)
    west  = find_west_connectivity(region, partition, global_grid)
    east  = find_east_connectivity(region, partition, global_grid)
    north = find_north_connectivity(region, partition, global_grid)
    south = find_south_connectivity(region, partition, global_grid)

    return (; west, east, north, south)
end

find_north_connectivity(region, ::XPartition, global_grid) = nothing

find_south_connectivity(region, ::XPartition, global_grid) = nothing

function find_east_connectivity(region, p::XPartition, global_grid)
    topo = topology(global_grid)
    if region == length(p)
        connectivity = topo[1] <: Periodic ? RegionalConnectivity(region, 1, East(), West()) : nothing
    else
        connectivity = RegionalConnectivity(region, region + 1, East(), West())
    end

    return connectivity
end

function find_west_connectivity(region, p::XPartition, global_grid)
    topo = topology(global_grid)

    if region == 1
        connectivity = topo[1] <: Periodic ? RegionalConnectivity(region, length(p), West(), East()) : nothing
    else
        connectivity = RegionalConnectivity(region, region - 1, West(), East())
    end

    return connectivity
end

find_east_connectivity(region, ::YPartition, global_grid) = nothing

find_west_connectivity(region, ::YPartition, global_grid) = nothing

function find_south_connectivity(region, p::YPartition, global_grid)
    topo = topology(global_grid)

    if region == 1
        connectivity = topo[1] <: Periodic ? RegionalConnectivity(region, length(p), South(), North()) : nothing
    else
        connectivity = RegionalConnectivity(region, region - 1, South(), North())
    end

    return connectivity
end

function find_north_connectivity(region, p::YPartition, global_grid)
    topo = topology(global_grid)

    if region == length(p)
        connectivity = topo[1] <: Periodic ? RegionalConnectivity(region, 1, North(), South()) : nothing
    else
        connectivity = RegionalConnectivity(region, region + 1, North(), South())
    end

    return connectivity
end
