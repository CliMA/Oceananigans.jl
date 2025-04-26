using Oceananigans.Architectures
using Oceananigans.Grids: topology, validate_tupled_argument
using CUDA: ndevices, device!

import Oceananigans.Architectures: device, cpu_architecture, on_architecture, array_type, child_architecture, convert_to_device
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

`x`, `y` and `z` can be:
- `x::Int`: allocate `x` processors to the first dimension
- `Equal()`: divide the domain in `x` equally among the remaining processes (not supported for multiple directions)
- `Fractional(ϵ₁, ϵ₂, ..., ϵₙ):` divide the domain unequally among `N` processes. The total work is `W = sum(ϵᵢ)`,
                                 and each process is then allocated `ϵᵢ / W` of the domain.
- `Sizes(n₁, n₂, ..., nₙ)`: divide the domain unequally. The total work is `W = sum(nᵢ)`,
                            and each process is then allocated `nᵢ`.

Examples:
========

```jldoctest
julia> using Oceananigans; using Oceananigans.DistributedComputations

julia> Partition(1, 4)
Partition across 4 = 1×4×1 ranks:
└── y: 4

julia> Partition(x = Fractional(1, 2, 3, 4))
Partition across 4 = 4×1×1 ranks:
└── x: Fractional(0.1, 0.2, 0.3, 0.4)

```
"""
Partition(x)    = Partition(validate_partition(x, nothing, nothing)...)
Partition(x, y) = Partition(validate_partition(x, y, nothing)...)

Partition(; x = nothing, y = nothing, z = nothing) = Partition(validate_partition(x, y, z)...)

function Base.show(io::IO, p::Partition)
    r = Rx, Ry, Rz = ranks(p)
    Nr = prod(r)
    last_rank = Nr - 1

    rank_info = if Nr == 1
        "1 rank"
    else
        "$Nr = $Rx×$Ry×$Rz ranks:"
    end

    print(io, "Partition across ", rank_info)

    if Rx > 1
        s = spine(Ry, Rz)
        print(io, '\n')
        print(io, s, " x: ", p.x)
    end

    if Ry > 1
        s = spine(Rz)
        print(io, '\n')
        print(io, s, " y: ", p.y)
    end

    if Rz > 1
        s = "└── "
        print(io, '\n')
        print(io, s, " z: ", p.z)
    end
end

spine(ξ, η=1) = ξ > 1 || η > 1 ? "├──" : "└──"

"""
    Equal()

Return a type that partitions a direction equally among remaining processes.

`Equal()` can be used for only one direction. Other directions must either be unspecified, or
specifically defined by `Int`, `Fractional`, or `Sizes`.
"""
struct Equal end

struct Fractional{S}
    sizes :: S
end

struct Sizes{S}
    sizes :: S
end

"""
    Fractional(ϵ₁, ϵ₂, ..., ϵₙ)

Return a type that partitions a direction unequally. The total work is `W = sum(ϵᵢ)`,
and each process is then allocated `ϵᵢ / W` of the domain.
"""
Fractional(args...) = Fractional(tuple(args ./ sum(args)...))  # We need to make sure that `sum(R) == 1`

"""
    Sizes(n₁, n₂, ..., nₙ)

Return a type that partitions a direction unequally. The total work is `W = sum(nᵢ)`,
and each process is then allocated `nᵢ`.
"""
Sizes(args...) = Sizes(tuple(args...))

Partition(x::Equal, y, z) = Partition(validate_partition(x, y, z)...)
Partition(x, y::Equal, z) = Partition(validate_partition(x, y, z)...)
Partition(x, y, z::Equal) = Partition(validate_partition(x, y, z)...)

Base.summary(s::Sizes)      = string("Sizes", s.sizes)
Base.summary(f::Fractional) = string("Fractional", f.sizes)

Base.show(io::IO, s::Sizes)      = print(io, summary(s))
Base.show(io::IO, f::Fractional) = print(io, summary(f))

ranks(p::Partition)  = (ranks(p.x), ranks(p.y), ranks(p.z))
ranks(::Nothing)     = 1 # a direction not partitioned fits in 1 rank
ranks(r::Int)        = r
ranks(r::Sizes)      = length(r.sizes)
ranks(r::Fractional) = length(r.sizes)

Base.size(p::Partition) = ranks(p)

# If a direction has only 1 rank, then it is not partitioned
validate_partition(x) = ifelse(ranks(x) == 1, nothing, x)

validate_partition(x, y, z) = map(validate_partition, (x, y, z))
validate_partition(::Equal, y, z) = remaining_workers(y, z), y, z

validate_partition(x, ::Equal, z) = x, remaining_workers(x, z), z
validate_partition(x, y, ::Equal) = x, y, remaining_workers(x, y)

function remaining_workers(r1, r2)
    MPI.Initialized() || MPI.Init()
    r12 = ranks(r1) * ranks(r2)
    return MPI.Comm_size(MPI.COMM_WORLD) ÷ r12
end

struct Distributed{A, S, Δ, R, ρ, I, C, γ, M, T, D} <: AbstractArchitecture
    child_architecture :: A
    partition :: Δ
    ranks :: R
    local_rank :: ρ
    local_index :: I
    connectivity :: C
    communicator :: γ
    mpi_requests :: M
    mpi_tag :: T
    devices :: D

    Distributed{S}(child_architecture :: A,
                   partition :: Δ,
                   ranks :: R,
                   local_rank :: ρ,
                   local_index :: I,
                   connectivity :: C,
                   communicator :: γ,
                   mpi_requests :: M,
                   mpi_tag :: T,
                   devices :: D) where {S, A, Δ, R, ρ, I, C, γ, M, T, D} =
                   new{A, S, Δ, R, ρ, I, C, γ, M, T, D}(child_architecture,
                                                        partition,
                                                        ranks,
                                                        local_rank,
                                                        local_index,
                                                        connectivity,
                                                        communicator,
                                                        mpi_requests,
                                                        mpi_tag,
                                                        devices)
end

#####
##### Constructors
#####

"""
    Distributed(child_architecture = CPU();
                partition = Partition(MPI.Comm_size(communicator)),
                devices = nothing,
                communicator = MPI.COMM_WORLD,
                synchronized_communication = false)

Return a distributed architecture that uses MPI for communications.

Positional arguments
====================

- `child_architecture`: Specifies whether the computation is performed on CPUs or GPUs.
                        Default: `CPU()`.

Keyword arguments
=================

- `partition`: A [`Partition`](@ref) specifying the total processors in the `x`, `y`, and `z` direction.
               Note that support for distributed `z` direction is  limited; we strongly suggest
               using partition with `z = 1` kwarg.

- `devices`: `GPU` device linked to local rank. The GPU will be assigned based on the
             local node rank as such `devices[node_rank]`. Make sure to run `--ntasks-per-node` <= `--gres=gpu`.
             If `nothing`, the devices will be assigned automatically based on the available resources.
             This argument is irrelevant if `child_architecture = CPU()`.

- `communicator`: the MPI communicator that orchestrates data transfer between nodes.
                  Default: `MPI.COMM_WORLD`.

- `synchronized_communication`: This keyword argument can be used to control downstream code behavior.
                                If `true`, then downstream code may use this tag to toggle between an algorithm
                                that permits communication between nodes "asynchronously" with other computations,
                                and an alternative serial algorithm in which communication and computation are
                                "synchronous" (that is, performed one after the other).
                                Default: `false`, specifying the use of asynchronous algorithms where supported,
                                which may result in faster time-to-solution.
"""
function Distributed(child_architecture = CPU();
                     partition = nothing,
                     devices = nothing,
                     communicator = nothing,
                     synchronized_communication = false)

    if !(MPI.Initialized())
        @info "MPI has not been initialized, so we are calling MPI.Init()."
        MPI.Init()
    end

    if isnothing(communicator) # default communicator
        communicator = MPI.COMM_WORLD
    end

    mpi_ranks = MPI.Comm_size(communicator)

    if isnothing(partition) # default partition
        partition = Partition(mpi_ranks)
    end

    ranks = Rx, Ry, Rz = size(partition)
    partition_ranks = Rx * Ry * Rz

    # TODO: make this error refer to `partition` (user input) rather than `ranks`
    if partition_ranks != mpi_ranks
        throw(ArgumentError("Partition($Rx, $Ry, $Rz) [$partition_ranks ranks] inconsistent " *
                            "with $mpi_ranks MPI ranks"))
    end

    local_rank         = MPI.Comm_rank(communicator)
    local_index        = rank2index(local_rank, Rx, Ry, Rz)
    # The rank connectivity _ALWAYS_ wraps around (The cartesian processor "grid" is `Periodic`)
    local_connectivity = NeighboringRanks(local_index, ranks)

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
                                                   Ref(0),
                                                   devices)
end

const DistributedCPU = Distributed{CPU}
const DistributedGPU = Distributed{GPU}

const SynchronizedDistributed = Distributed{<:Any, true}
const AsynchronousDistributed = Distributed{<:Any, false}

#####
##### All the architectures
#####

ranks(arch::Distributed) = ranks(arch.partition)

child_architecture(arch::Distributed) = arch.child_architecture
device(arch::Distributed)             = device(child_architecture(arch))

zeros(arch::Distributed, FT, N...)         = zeros(child_architecture(arch), FT, N...)
array_type(arch::Distributed)              = array_type(child_architecture(arch))
sync_device!(arch::Distributed)            = sync_device!(arch.child_architecture)
convert_to_device(arch::Distributed, arg)  = convert_to_device(child_architecture(arch), arg)

# Switch to a synchronized architecture
synchronized(arch) = arch
synchronized(arch::Distributed) = Distributed{true}(child_architecture(arch),
                                                    arch.partition,
                                                    arch.ranks,
                                                    arch.local_rank,
                                                    arch.local_index,
                                                    arch.connectivity,
                                                    arch.communicator,
                                                    arch.mpi_requests,
                                                    arch.mpi_tag,
                                                    arch.devices)

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
                   arch.mpi_tag,
                   nothing) # No devices on the CPU

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

mutable struct NeighboringRanks{E, W, N, S, SW, SE, NW, NE}
         east :: E
         west :: W
        north :: N
        south :: S
    southwest :: SW
    southeast :: SE
    northwest :: NW
    northeast :: NE
end

const NoConnectivity = NeighboringRanks{Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing}

"""
    NeighboringRanks(; east, west, north, south, southwest, southeast, northwest, northeast)

Generate a `NeighboringRanks` object that holds the MPI ranks of the neighboring processors.
"""
NeighboringRanks(; east, west, north, south, southwest, southeast, northwest, northeast) =
    NeighboringRanks(east, west, north, south, southwest, southeast, northwest, northeast)

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

function NeighboringRanks(local_index, ranks)
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

    return NeighboringRanks(west=west_rank, east=east_rank,
                            south=south_rank, north=north_rank,
                            southwest=southwest_rank,
                            southeast=southeast_rank,
                            northwest=northwest_rank,
                            northeast=northeast_rank)
end

#####
##### Pretty printing
#####

function Base.summary(arch::Distributed)
    child_arch = child_architecture(arch)
    A = typeof(child_arch)
    return string("Distributed{$A}")
end

function Base.show(io::IO, arch::Distributed)

    Rx, Ry, Rz = arch.ranks
    local_rank = arch.local_rank
    Nr = prod(arch.ranks)
    last_rank = Nr - 1

    rank_info = if last_rank == 0
        "1 rank:"
    else
        "$Nr = $Rx×$Ry×$Rz ranks:"
    end

    print(io, summary(arch), " across ", rank_info, '\n')
    print(io, "├── local_rank: ", local_rank, " of 0-$last_rank", '\n')

    ix, iy, iz = arch.local_index
    index_info = string("index [$ix, $iy, $iz]")

    c = arch.connectivity
    connectivity_info = if c isa NeighboringRanks
        string("└── connectivity:",
               isnothing(c.east)      ? "" : " east=$(c.east)",
               isnothing(c.west)      ? "" : " west=$(c.west)",
               isnothing(c.north)     ? "" : " north=$(c.north)",
               isnothing(c.south)     ? "" : " south=$(c.south)",
               isnothing(c.southwest) ? "" : " southwest=$(c.southwest)",
               isnothing(c.southeast) ? "" : " southeast=$(c.southeast)",
               isnothing(c.northwest) ? "" : " northwest=$(c.northwest)",
               isnothing(c.northeast) ? "" : " northeast=$(c.northeast)")
    end

    if isnothing(connectivity_info)
        print(io, "└── local_index: [$ix, $iy, $iz]")
    else
        print(io, "├── local_index: [$ix, $iy, $iz]", '\n')
        print(io, connectivity_info)
    end
end

