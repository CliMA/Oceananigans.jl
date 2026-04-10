using Oceananigans.ImmersedBoundaries: AbstractImmersedBoundary, ImmersedBoundaryGrid, immersed_cell

# Partition that allows multiple neighbors. 
# We loose the "cartesian" nature of the partition, but just have ranks 0 ... Rmax
# where each rank is defined by its 4 edges in terms of (i, j) positions.
# With this partition, local_index does not make sense anymore...
struct UnstructuredRectangularPartition{R}
    south_west :: R # Vector{Tuple{Int, Int}} of length R
    north_west :: R # Vector{Tuple{Int, Int}} of length R
    north_east :: R # Vector{Tuple{Int, Int}} of length R
    south_east :: R # Vector{Tuple{Int, Int}} of length R
end

ranks(r::UnstructuredRectangularPartition) = length(r.south_west)
Base.size(r::UnstructuredRectangularPartition) = (ranks(r), 1, 1)

function NeighboringRanks(r::UnstructuredRectangularPartition, local_index, ranks)

    # Start and end indices for each core
    west_indices  = getindex.(south_west, 1)
    east_indices  = getindex.(north_east, 1)
    south_indices = getindex.(south_west, 2)
    north_indices = getindex.(north_east, 2)

    # First we need to find the limits of the grid in x and y
    Nx = maximum(east_indices)
    Ny = maximum(north_indices)
    
    # Now my local i, j limits
    iw =  west_indices[local_index]
    ie =  east_indices[local_index]
    js = south_indices[local_index]
    jn = north_indices[local_index]

    # Now let's filter the cores to find the limits

    # starting from finding the targets
    iet = ifelse(iw == 1,  Nx, iw-1)
    iwt = ifelse(ie == Nx,  1, ie+1)
    jnt = ifelse(js == 1,  Ny, is-1)
    jst = ifelse(iw == Ny,  1, in+1)

    # Let's find possible cores that overlap, starting on the west boundary
    Rw = findall(west_indices  == iwt) # Minimum 1 core
    Re = findall(east_indices  == iet) # Minimum 1 core
    Rs = findall(south_indices == jst) # Minimum 1 core
    Rn = findall(north_indices == jnt) # Minimum 1 core

    # Now restrict these cores based on whether they overlap or not with our core
    # Note that we need to take also the cores next to them because we possibly might need
    # some corners. so let's give us a graph representation where all the neighbors are counted


    return NeighboringRanks(west=west_rank, east=east_rank,
                            south=south_rank, north=north_rank,
                            southwest=southwest_rank,
                            southeast=southeast_rank,
                            northwest=northwest_rank,
                            northeast=northeast_rank)
end
