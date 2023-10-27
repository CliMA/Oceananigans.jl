using Oceananigans.Architectures
using Oceananigans.Grids: topology, validate_tupled_argument
using CUDA: ndevices, device!

import Oceananigans.Architectures: device, cpu_architecture, arch_array, array_type, child_architecture
import Oceananigans.Grids: zeros
import Oceananigans.Utils: sync_device!, tupleit

import Base

#####
##### Partitioning
#####

struct Partition{Sx, Sy, Sz}
    x :: Sx
    y :: Sy
    z :: Sz
end

"""
    Partition(; x = 1, y = 1, z = 1)

Return `Partition` representing the division of a domain in
the `x` (first), `y` (second) and `z` (third) dimension

Keyword arguments: 
==================

- `x`: partitioning of the first dimension 
- `y`: partitioning of the second dimension
- `z`: partitioning of the third dimension

if supplied as positional arguments `x` will be the first argument, 
`y` the second and `z` the third

`x`, `y` and `z` can be `Int`, `Equal`, `Fractional` or `Sizes` 
(see below)

Examples:
========

```jldoctest
julia> using Oceananigans; using Oceananigans.DistributedComputations

julia> Partition(1, 4)
Domain partitioning with (1, 4, 1) ranks
├── x-partitioning: none
├── y-partitioning: 4
└── z-partitioning: none

julia> Partition(x = Fractional(1, 2, 3, 4))
Domain partitioning with (4, 1, 1) ranks
├── x-partitioning: domain fractions: (0.1, 0.2, 0.3, 0.4)
├── y-partitioning: none
└── z-partitioning: none

```
"""
Partition(x)       = Partition(validate_partition(x, 1, 1)...)
Partition(x, y)    = Partition(validate_partition(x, y, 1)...)

Partition(; x = 1, y = 1, z = 1) = Partition(validate_partition(x, y, z)...)

Base.show(io::IO, p::Partition) =
    print(io,
    "Domain partitioning with $(ranks(p)) ranks", "\n",
    "├── x-partitioning: $(ranks(p.x) == 1 ? "none" : p.x)", "\n",
    "├── y-partitioning: $(ranks(p.y) == 1 ? "none" : p.y)", "\n",
    "└── z-partitioning: $(ranks(p.z) == 1 ? "none" : p.z)")

"""type representing equal domain partitioning (not supported for more than one direction)"""
struct Equal end

"""type representing fractional domain partioning where rank `1` holds `sizes[1]` parts of the domain"""
struct Fractional{S} 
    sizes :: S
end

"""type representing domain partioning where rank `1` holds `sizes[1]` grid cells"""
struct Sizes{S} 
    sizes :: S
end

Partition(x::Equal, y, z) = Partition(validate_partition(x, y, z)...)
Partition(x, y::Equal, z) = Partition(validate_partition(x, y, z)...)
Partition(x, y, z::Equal) = Partition(validate_partition(x, y, z)...)

Base.show(io::IO, s::Sizes)      = print(io, "domain sizes:     $(s.sizes)")
Base.show(io::IO, s::Fractional) = print(io, "domain fractions: $(s.sizes)")

ranks(p::Partition)  = (ranks(p.x), ranks(p.y), ranks(p.z))
ranks(r::Int)        = r
ranks(r::Sizes)      = length(r.sizes)
ranks(r::Fractional) = length(r.sizes)

Base.size(p::Partition) = ranks(p)

Fractional(args...) = Fractional(tuple(args ./ sum(args)...))  # We need to make sure that `sum(R) == 1`
     Sizes(args...) = Sizes(tuple(args...))

validate_partition(x, y, z) = (x, y, z)
validate_partition(::Equal, y, z) = remaining_workers(y, z), y, z

validate_partition(x, ::Equal, z) = x, remaining_workers(x, z), z
validate_partition(x, y, ::Equal) = x, y, remaining_workers(x, y)

function remaining_workers(r1, r2)
    MPI.Initialized() || MPI.Init()    
    return MPI.Comm_size(MPI.COMM_WORLD) ÷ ranks(r1)*ranks(r2)
end

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
cpu_architecture(arch::Distributed{A, S}) where {A, S} = 
    Distributed{S}(CPU(),
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
              