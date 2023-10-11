using Oceananigans.Architectures
using Oceananigans.Grids: topology, validate_tupled_argument
using CUDA: ndevices, device!

import Oceananigans.Architectures: device, arch_array, array_type, child_architecture
import Oceananigans.Grids: zeros
import Oceananigans.Utils: sync_device!, tupleit

#####
##### Partitioning
#####

struct Partition{Sx, Sy, Sz}
    Rx :: Sx
    Ry :: Sy
    Rz :: Sz
end

"""
    Partition(Rx::Number, Ry::Number=1, Rz::Number=1)

Return `Partition` representing the division of a domain into
`Rx` parts in `x` and `Ry` parts in `y` and `Rz` parts in `z`,
where `x, y, z` are the first, second, and third dimension
respectively.

`Rx`, `Ry` and `Rz` can be vectors containing the percentage of the domain
ascribed to the different cores. In this case `length(Rx)` represent the number
of divisions in the first dimension with `sum(Rx)` equal to 1.0. 
Rank `arch.local_index[i]` will have `global_size[i] * Rx[i]` elements in the first dimension.
"""
Partition(Rx)     = Partition(validate_partition(Rx), 1, 1)
Partition(Rx, Ry) = Partition(validate_partition(Rx), validate_partition(Ry), 1)

function Partition(; Rx = 1, Ry = 1, Rz = 1)
    Rx = validate_partition(Rx)
    Ry = validate_partition(Ry)
    Rz = validate_partition(Rz)
    return Partition(Rx, Ry, Rz)
end

# We need to make sure that the domain is partitioned correctly in percentages, i.e that `sum(R) == 1`
validate_partition(r::Number) = r
validate_partition(r::AbstractVector) = sum(r) == 1 ? r : throw(ArgumentError("The sum of the partition must be 1.0"))

ranks(p::Partition) = (p.Rx, p.Ry, p.Rz)

struct Distributed{A, S, Δ, R, ρ, I, C, γ, M, T} <: AbstractArchitecture
    child_architecture :: A
    partition :: Δ
    ranks :: R
    local_rank :: ρ
    local_index :: I
    connectivity :: C
    communicator :: γ
    mpi_requests :: M
    mpi_tag :: T

    Distributed{S}(child_architecture :: A,
                   partition :: Δ,
                   ranks :: R,
                   local_rank :: ρ,
                   local_index :: I,
                   connectivity :: C,
                   communicator :: γ,
                   mpi_requests :: M,
                   mpi_tag :: T) where {S, A, Δ, R, ρ, I, C, γ, M, T} = 
                   new{A, S, Δ, R, ρ, I, C, γ, M, T}(child_architecture,
                                                     partition,
                                                     ranks,
                                                     local_rank,
                                                     local_index,
                                                     connectivity,
                                                     communicator,
                                                     mpi_requests,
                                                     mpi_tag)
end

#####
##### Constructors
#####

"""
    Distributed(child_architecture = CPU(); 
                topology, 
                partition,
                devices = nothing, 
                communicator = MPI.COMM_WORLD)

Constructor for a distributed architecture that uses MPI for communications

Positional arguments
=================

- `child_architecture`: Specifies whether the computation is performed on CPUs or GPUs. 
                        Default: `child_architecture = CPU()`.

Keyword arguments
=================
                        
- `synchronized_communication`: if true, always use synchronized communication through ranks

- `ranks` (required): A 3-tuple `(Rx, Ry, Rz)` specifying the total processors in the `x`, 
                      `y` and `z` direction. NOTE: support for distributed z direction is 
                      limited, so `Rz = 1` is strongly suggested.

- `devices`: `GPU` device linked to local rank. The GPU will be assigned based on the 
             local node rank as such `devices[node_rank]`. Make sure to run `--ntasks-per-node` <= `--gres=gpu`.
             If `nothing`, the devices will be assigned automatically based on the available resources

- `communicator`: the MPI communicator, `MPI.COMM_WORLD`. This keyword argument should not be tampered with 
                  if not for testing or developing. Change at your own risk!
"""
function Distributed(child_architecture = CPU(); 
                     communicator = MPI.COMM_WORLD,
                     devices = nothing, 
                     synchronized_communication = false,
                     partition = Partition(MPI.Comm_size(communicator)))

    if !(MPI.Initialized())
        @info "MPI has not been initialized, so we are calling MPI.Init()".
        MPI.Init()
    end

    ranks = size(partition)
    Rx, Ry, Rz = ranks
    total_ranks = Rx * Ry * Rz
    mpi_ranks  = MPI.Comm_size(communicator)

    # TODO: make this error refer to `partition` (user input) rather than `ranks`
    if total_ranks != mpi_ranks
        throw(ArgumentError("Partition($Rx, $Ry, $Rz) [$total_ranks total ranks] inconsistent " *
                            "with number of MPI ranks: $mpi_ranks."))
    end
    
    local_rank         = MPI.Comm_rank(communicator)
    local_index        = rank2index(local_rank, Rx, Ry, Rz)
    # The rank connectivity _ALWAYS_ wraps around (The cartesian processor "grid" is `Periodic`)
    local_connectivity = RankConnectivity(local_index, ranks) 

    # Assign CUDA device if on GPUs
    if child_architecture isa GPU
        local_comm = MPI.Comm_split_type(communicator, MPI.COMM_TYPE_SHARED, local_rank)
        node_rank  = MPI.Comm_rank(local_comm)
        isnothing(devices) ? device!(node_rank % ndevices()) : device!(devices[node_rank+1]) 
    end

    mpi_requests = MPI.Request[]

    return Distributed{synchronized_communication}(child_architecture,
                                                   partition,
                                                   ranks,
                                                   local_rank,
                                                   local_index,
                                                   local_connectivity,
                                                   communicator,
                                                   mpi_requests,
                                                   Ref(0))
end

const DistributedCPU = Distributed{CPU}
const DistributedGPU = Distributed{GPU}

const SynchronizedDistributed = Distributed{<:Any, true}

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
cpu_architecture(arch::DistributedGPU) = Distributed(CPU(),
                                                     arch.partition, 
                                                     arch.ranks, 
                                                     arch.local_rank,
                                                     arch.local_index,
                                                     arch.connectivity,
                                                     arch.communicator,
                                                     arch.mpi_requests,
                                                     arch.mpi_tag)

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
function increment_index(i, R)
    R == 1 && return nothing
    if i+1 > R
        return 1
    else
        return i+1
    end
end

function decrement_index(i, R)
    R == 1 && return nothing
    if i-1 < 1
        return R
    else
        return i-1
    end
end

function RankConnectivity(local_index, ranks)
    i, j, k = local_index
    Rx, Ry, Rz = ranks

    i_east  = increment_index(i, Rx)
    i_west  = decrement_index(i, Rx)
    j_north = increment_index(j, Ry)
    j_south = decrement_index(j, Ry)

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
              