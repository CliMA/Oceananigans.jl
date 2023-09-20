using Oceananigans.Architectures
using Oceananigans.Grids: topology, validate_tupled_argument
using CUDA: ndevices, device!

import Oceananigans.Architectures: device, arch_array, array_type, child_architecture
import Oceananigans.Grids: zeros
import Oceananigans.Utils: sync_device!

struct Distributed{A, M, R, I, ρ, C, γ, T} <: AbstractArchitecture
  child_architecture :: A
          local_rank :: R
         local_index :: I
               ranks :: ρ
        connectivity :: C
        communicator :: γ
        mpi_requests :: M
             mpi_tag :: T
end

#####
##### Constructors
#####

"""
    Distributed(child_architecture = CPU(); 
                    topology, 
                    ranks, 
                    devices = nothing, 
                    communicator = MPI.COMM_WORLD)

Constructor for a distributed architecture that uses MPI for communications


Positional arguments
=================

- `child_architecture`: Specifies whether the computation is performed on CPUs or GPUs. 
                        Default: `child_architecture = CPU()`.

Keyword arguments
=================

- `topology` (required): the topology we want the grid to have. It is used to establish connectivity.
                        
- `ranks` (required): A 3-tuple `(Rx, Ry, Rz)` specifying the total processors in the `x`, 
                      `y` and `z` direction. NOTE: support for distributed z direction is 
                      limited, so `Rz = 1` is strongly suggested.

- enable_overlapped_computation: if `true` the prognostic halo communication will be overlapped
                                 with tendency calculations, and the barotropic halo communication
                                 with the implicit vertical solver (defaults to `true`)

- `devices`: `GPU` device linked to local rank. The GPU will be assigned based on the 
             local node rank as such `devices[node_rank]`. Make sure to run `--ntasks-per-node` <= `--gres=gpu`.
             If `nothing`, the devices will be assigned automatically based on the available resources

- `communicator`: the MPI communicator, `MPI.COMM_WORLD`. This keyword argument should not be tampered with 
                  if not for testing or developing. Change at your own risk!
"""
function Distributed(child_architecture = CPU(); 
                         topology, 
                         ranks,
                         devices = nothing, 
                         enable_overlapped_computation = true,
                         communicator = MPI.COMM_WORLD)

    MPI.Initialized() || error("Must call MPI.Init() before constructing a MultiCPU.")

    validate_tupled_argument(ranks, Int, "ranks")

    Rx, Ry, Rz = ranks
    total_ranks = Rx*Ry*Rz

    mpi_ranks  = MPI.Comm_size(communicator)
    local_rank = MPI.Comm_rank(communicator)

    if total_ranks != mpi_ranks
        throw(ArgumentError("ranks=($Rx, $Ry, $Rz) [$total_ranks total] inconsistent " *
                            "with number of MPI ranks: $mpi_ranks."))
    end
    
    local_index        = rank2index(local_rank, Rx, Ry, Rz)
    local_connectivity = RankConnectivity(local_index, ranks, topology)

    A = typeof(child_architecture)
    R = typeof(local_rank)    
    I = typeof(local_index)   
    ρ = typeof(ranks)         
    C = typeof(local_connectivity)  
    γ = typeof(communicator)  

    # Assign CUDA device if on GPUs
    if child_architecture isa GPU
        local_comm = MPI.Comm_split_type(communicator, MPI.COMM_TYPE_SHARED, local_rank)
        node_rank  = MPI.Comm_rank(local_comm)
        isnothing(devices) ? device!(node_rank % ndevices()) : device!(devices[node_rank+1]) 
    end

    mpi_requests = enable_overlapped_computation ? MPI.Request[] : nothing

    M = typeof(mpi_requests)
    T = typeof(Ref(0))

    return Distributed{A, M, R, I, ρ, C, γ, T}(child_architecture, local_rank, local_index, ranks, local_connectivity, communicator, mpi_requests, Ref(0))
end

const DistributedCPU = Distributed{CPU}
const DistributedGPU = Distributed{GPU}

const BlockingDistributed = Distributed{<:Any, <:Nothing}

#####
##### All the architectures
#####

child_architecture(arch::Distributed) = arch.child_architecture
device(arch::Distributed)             = device(child_architecture(arch))
arch_array(arch::Distributed, A)      = arch_array(child_architecture(arch), A)
zeros(FT, arch::Distributed, N...)    = zeros(FT, child_architecture(arch), N...)
array_type(arch::Distributed)         = array_type(child_architecture(arch))
sync_device!(arch::Distributed)       = sync_device!(arch.child_architecture)

cpu_architecture(arch::DistributedCPU) = arch
cpu_architecture(arch::DistributedGPU) = 
    Distributed(CPU(), arch.local_rank, arch.local_index, arch.ranks, 
                           arch.connectivity, arch.communicator, arch.mpi_requests, arch.mpi_tag)

#####
##### Converting between index and MPI rank taking k as the fast index
#####

index2rank(i, j, k, Rx, Ry, Rz) = (i-1)*Ry*Rz + (j-1)*Rz + (k-1)

function rank2index(r, Rx, Ry, Rz)
    i = div(r, Ry*Rz)
    r -= i*Ry*Rz
    j = div(r, Rz)
    k = mod(r, Rz)
    return i+1, j+1, k+1  # 1-based Julia
end

#####
##### Rank connectivity graph
#####

struct RankConnectivity{E, W, N, S, SW, SE, NW, NE}
         east :: E
         west :: W
        north :: N
        south :: S
    southwest :: SW
    southeast :: SE
    northwest :: NW
    northeast :: NE
end

"""
    RankConnectivity(; east, west, north, south, southwest, southeast, northwest, northeast)

generate a `RankConnectivity` object that holds the MPI ranks of the neighboring processors.
"""
RankConnectivity(; east, west, north, south, southwest, southeast, northwest, northeast) =
    RankConnectivity(east, west, north, south, southwest, southeast, northwest, northeast)

# The "Periodic" topologies are `Periodic`, `FullyConnected` and `RightConnected`
# The "Bounded" topologies are `Bounded` and `LeftConnected`

function increment_index(i, R, topo)
    R == 1 && return nothing
    if i+1 > R
        if topo == Periodic || topo == FullyConnected || topo == RightConnected
            return 1
        else
            return nothing
        end
    else
        return i+1
    end
end

function decrement_index(i, R, topo)
    R == 1 && return nothing
    if i-1 < 1
        if topo == Periodic || topo == FullyConnected || topo == RightConnected
            return R
        else
            return nothing
        end
    else
        return i-1
    end
end

function RankConnectivity(local_index, ranks, topology)
    i, j, k = local_index
    Rx, Ry, Rz = ranks
    TX, TY, TZ = topology

    i_east  = increment_index(i, Rx, TX)
    i_west  = decrement_index(i, Rx, TX)
    j_north = increment_index(j, Ry, TY)
    j_south = decrement_index(j, Ry, TY)

     east_rank = isnothing(i_east)  ? nothing : index2rank(i_east,  j, k, Rx, Ry, Rz)
     west_rank = isnothing(i_west)  ? nothing : index2rank(i_west,  j, k, Rx, Ry, Rz)
    north_rank = isnothing(j_north) ? nothing : index2rank(i, j_north, k, Rx, Ry, Rz)
    south_rank = isnothing(j_south) ? nothing : index2rank(i, j_south, k, Rx, Ry, Rz)

    northeast_rank = isnothing(i_east) || isnothing(j_north) ? nothing : index2rank(i_east, j_north, k, Rx, Ry, Rz)
    northwest_rank = isnothing(i_west) || isnothing(j_north) ? nothing : index2rank(i_west, j_north, k, Rx, Ry, Rz)
    southeast_rank = isnothing(i_east) || isnothing(j_south) ? nothing : index2rank(i_east, j_south, k, Rx, Ry, Rz)
    southwest_rank = isnothing(i_west) || isnothing(j_south) ? nothing : index2rank(i_west, j_south, k, Rx, Ry, Rz)

    return RankConnectivity(west=west_rank, east=east_rank, 
                            south=south_rank, north=north_rank,
                            southwest=southwest_rank,
                            southeast=southeast_rank,
                            northwest=northwest_rank,
                            northeast=northeast_rank)
end

#####
##### Pretty printing
#####

function Base.show(io::IO, arch::Distributed)
    c = arch.connectivity
    print(io, "Distributed architecture (rank $(arch.local_rank)/$(prod(arch.ranks)-1)) [index $(arch.local_index) / $(arch.ranks)]\n",
              "└── child architecture: $(typeof(child_architecture(arch))) \n",
              "└── connectivity:",
              isnothing(c.east) ? "" : " east=$(c.east)",
              isnothing(c.west) ? "" : " west=$(c.west)",
              isnothing(c.north) ? "" : " north=$(c.north)",
              isnothing(c.south) ? "" : " south=$(c.south)",
              isnothing(c.southwest) ? "" : " southwest=$(c.southwest)",
              isnothing(c.southeast) ? "" : " southeast=$(c.southeast)",
              isnothing(c.northwest) ? "" : " northwest=$(c.northwest)",
              isnothing(c.northeast) ? "" : " northeast=$(c.northeast)")
end
              
