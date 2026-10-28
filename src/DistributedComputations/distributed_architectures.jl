using Oceananigans.Architectures
using Oceananigans.Grids: topology

import Oceananigans.Architectures: device, device!, cpu_architecture, on_architecture, array_type, child_architecture, convert_to_device
import Oceananigans.Grids: zeros
import Oceananigans.Utils: sync_device!

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

A tuple of these in `x` (or `y`) divides every row (or column) of ranks separately, so that ranks do not have to
line up across rows (or columns). All rows (or columns) hold the same number of ranks.

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

julia> using Oceananigans.DistributedComputations: Sizes

julia> Partition(x = 2, y = (Sizes(10, 22), Sizes(20, 12)))
Partition across 4 = 2×2×1 ranks:
├── x: 2
└── y: (Sizes(10, 22), Sizes(20, 12))

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
Partition(x::Equal, y::Equal, z) = Partition(validate_partition(x, y, z)...)
Partition(x::Equal, y, z::Equal) = Partition(validate_partition(x, y, z)...)
Partition(x, y::Equal, z::Equal) = Partition(validate_partition(x, y, z)...)
Partition(x::Equal, y::Equal, z::Equal) = Partition(validate_partition(x, y, z)...)

Base.summary(s::Sizes)      = string("Sizes", s.sizes)
Base.summary(f::Fractional) = string("Fractional", f.sizes)

Base.show(io::IO, s::Sizes)      = print(io, summary(s))
Base.show(io::IO, f::Fractional) = print(io, summary(f))

ranks(p::Partition)  = (ranks(p.x), ranks(p.y), ranks(p.z))
ranks(::Nothing)     = 1 # a direction not partitioned fits in 1 rank
ranks(r::Int)        = r
ranks(r::Sizes)      = length(r.sizes)
ranks(r::Fractional) = length(r.sizes)
ranks(r::Tuple)      = ranks(first(r))

Base.size(p::Partition) = ranks(p)

# If a direction has only 1 rank, then it is not partitioned
validate_partition(x) = ifelse(ranks(x) == 1, nothing, x)

validate_partition(x, y, z) = map(validate_partition, (x, y, z))
validate_partition(::Equal, y, z) = remaining_workers(y, z), y, z

validate_partition(x, ::Equal, z) = x, remaining_workers(x, z), z
validate_partition(x, y, ::Equal) = x, y, remaining_workers(x, y)

validate_partition(::Equal, ::Equal, z) = throw_multiple_equal()
validate_partition(::Equal, y, ::Equal) = throw_multiple_equal()
validate_partition(x, ::Equal, ::Equal) = throw_multiple_equal()
validate_partition(::Equal, ::Equal, ::Equal) = throw_multiple_equal()
throw_multiple_equal() = throw(ArgumentError("Equal() can be used for only one direction"))

function remaining_workers(r1, r2)
    MPI.Initialized() || MPI.Init(; threadlevel = has_spare_thread() ? :multiple : :serialized)
    r12 = ranks(r1) * ranks(r2)
    return MPI.Comm_size(MPI.COMM_WORLD) ÷ r12
end

struct Distributed{A, S, Δ, R, ρ, I, C, γ, T, D} <: AbstractArchitecture
    child_architecture :: A
    partition :: Δ
    ranks :: R
    local_rank :: ρ
    local_index :: I
    connectivity :: C
    communicator :: γ
    field_count :: T
    devices :: D

    Distributed{S}(child_architecture :: A,
                   partition :: Δ,
                   ranks :: R,
                   local_rank :: ρ,
                   local_index :: I,
                   connectivity :: C,
                   communicator :: γ,
                   field_count :: T,
                   devices :: D) where {S, A, Δ, R, ρ, I, C, γ, T, D} =
                   new{A, S, Δ, R, ρ, I, C, γ, T, D}(child_architecture,
                                                        partition,
                                                        ranks,
                                                        local_rank,
                                                        local_index,
                                                        connectivity,
                                                        communicator,
                                                        field_count,
                                                        devices)
end

#####
##### Constructors
#####

"""
    Distributed(child_architecture = CPU();
                partition = nothing,
                devices = nothing,
                communicator = nothing,
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
             If `nothing` (default), the devices will be assigned automatically based on the available resources.
             This argument is irrelevant if `child_architecture = CPU()`.

- `communicator`: the MPI communicator that orchestrates data transfer between nodes, e.g., `MPI.COMM_WORLD`.
                  Default: `nothing`.

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
                     synchronized_communication = false,
                     field_count = Threads.Atomic{UInt64}(0))

    if !(MPI.Initialized())
        @info "MPI has not been initialized, so we are calling MPI.Init()."
        MPI.Init(; threadlevel = has_spare_thread() ? :multiple : :serialized)
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

    if partition_ranks == 1
        @warn "We are building a Distributed architecture on a single MPI rank. \n" *
              "This can occur when MPI is incorrectly configured. \n" *
              "See https://juliaparallel.org/MPI.jl/stable/configuration/ for more details."
    end

    # TODO: make this error refer to `partition` (user input) rather than `ranks`
    if partition_ranks != mpi_ranks
        throw(ArgumentError("Partition($Rx, $Ry, $Rz) [$partition_ranks ranks] inconsistent " *
                            "with $mpi_ranks MPI ranks"))
    end

    local_rank  = MPI.Comm_rank(communicator)
    local_index = rank2index(local_rank, Rx, Ry, Rz)

    # Assign GPU device if on GPUs
    if child_architecture isa GPU
        local_comm = MPI.Comm_split_type(communicator, MPI.COMM_TYPE_SHARED, local_rank)
        node_rank  = MPI.Comm_rank(local_comm)
        isnothing(devices) ? device!(child_architecture, node_rank % ndevices(child_architecture)) : device!(child_architecture, devices[node_rank+1])
    end

    return Distributed{synchronized_communication}(child_architecture,
                                                   partition,
                                                   ranks,
                                                   local_rank,
                                                   local_index,
                                                   nothing,
                                                   communicator,
                                                   field_count,
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
                                                    arch.field_count,
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
                   arch.field_count,
                   nothing) # No devices on the CPU

# Fields of architectures sharing `field_count` (e.g. `twin_grid`s) get distinct tags, as long as all ranks create them in the same order
next_field_tag!(arch::Distributed) = Threads.atomic_add!(arch.field_count, UInt64(1))

# Fail loudly instead of hanging in the first halo exchange: called where all ranks meet, e.g. `initialize!(::Simulation)`
check_field_tags(arch) = nothing
check_field_tags(arch::Distributed) = check_field_tags(arch.communicator, arch.field_count[])
check_field_tags(communicator, field_count) = nothing # e.g. Reactant meshes

function check_field_tags(communicator::MPI.Comm, field_count)
    MPI.Allreduce(field_count, min, communicator) == MPI.Allreduce(field_count, max, communicator) ||
        throw(ArgumentError("MPI ranks created different numbers of distributed fields, so their halo exchanges would never match. " *
                            "All ranks must create the same distributed fields, in the same order."))
    return nothing
end

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
##### Rank connectivity
#####

"""
    RankConnectivity(partition, global_size, topology, halo)

Global index ranges of every rank (in rank order) on a grid of `global_size` and `topology` divided by `partition`.
In the directions that are partitioned, every rank must be at least as large as the `halo`.
"""
struct RankConnectivity{T}
    ranges :: Vector{NTuple{3, UnitRange{Int}}}
    size :: NTuple{3, Int}
    topology :: T
end

function RankConnectivity(partition, global_size, topology, halo)
    Rx, Ry, Rz = ranks(partition)
    Nx, Ny, Nz = global_size

    # The index ranges of the ranks that divide `N` cells according to `p`
    function index_ranges(N, p)
        n = [local_size(N, p, r) for r in 1:ranks(p)]
        return [sum(n[1:r-1]) + 1 : sum(n[1:r]) for r in eachindex(n)]
    end

    # A tuple partitions every row (in x) or column (in y) of ranks separately
    x = [index_ranges(Nx, partition.x isa Tuple ? partition.x[j] : partition.x) for j in 1:Ry]
    y = [index_ranges(Ny, partition.y isa Tuple ? partition.y[i] : partition.y) for i in 1:Rx]
    z = index_ranges(Nz, partition.z)

    ranges = map(0:Rx*Ry*Rz-1) do rank
        i, j, k = rank2index(rank, Rx, Ry, Rz)
        return (x[j][i], y[i][j], z[k])
    end

    for d in 1:3
        n = minimum(r -> length(r[d]), ranges)
        if n < global_size[d] && n < halo[d]
            throw(ArgumentError("halo=$(halo[d]) must be ≤ the size of every rank, found a rank of size $n in dimension $d"))
        end
    end

    return RankConnectivity(ranges, (Nx, Ny, Nz), topology)
end

#####
##### Pretty printing
#####

function Base.summary(arch::Distributed)
    child_arch = child_architecture(arch)
    A = Base.summary(child_arch)
    Rx, Ry, Rz = ranks(arch)
    return "Distributed{$A} on $(Rx)×$(Ry)×$(Rz)"
end

function Base.show(io::IO, arch::Distributed)

    Rx, Ry, Rz = arch.ranks
    local_rank = arch.local_rank
    Nr = prod(arch.ranks)
    last_rank = Nr - 1

    rank_info = if last_rank == 0
        "1 rank"
    else
        "$Nr = $Rx×$Ry×$Rz ranks"
    end

    sync_type = if arch isa SynchronizedDistributed
        "synchronous"
    elseif arch isa AsynchronousDistributed
        "asynchronous"
    else
        "unknown"
    end

    print(io, summary(arch), " across ", rank_info, ", with $sync_type communications:", '\n')
    print(io, "├── local_rank: ", local_rank, " of 0-$last_rank", '\n')

    ix, iy, iz = arch.local_index
    print(io, "└── local_index: [$ix, $iy, $iz]")
end
