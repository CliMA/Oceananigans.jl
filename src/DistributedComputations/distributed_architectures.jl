using Oceananigans.Architectures
using Oceananigans.Grids: topology, validate_tupled_argument
using CUDA: ndevices, device!

import Oceananigans.Architectures: device, arch_array, array_type, child_architecture
import Oceananigans.Grids: zeros
import Oceananigans.Utils: sync_device!, tupleit

#####
##### Partitioning
#####

# Form 3-tuple
regularize_size(sz::Tuple{<:Int, <:Int}, Rx, Ry) = (sz[1], sz[2], 1)

# Infer whether x- or y-dimension is indicated
function regularize_size(N::Int, Rx, Ry)
    if Rx == 1
        return (1, N, 1)
    elseif Ry == 1
        return (N, 1, 1)
    else
        throw(ArgumentError("We can't interpret the 1D size $N for 2D partitions!"))
    end
end

regularize_size(sz::Array{<:Int, 1}) = regularize_size(sz[1])
regularize_size(sz::Array{<:Int, 2}) = regularize_size((sz[1], sz[2]))
regularize_size(sz::Tuple{<:Int})    = regularize_size(sz[1])

# For 3D partitions:
# regularize_size(sz::Tuple{<:Int, <:Int, <:Int}, Rx, Ry) = sz

struct Partition{S, Rx, Ry, Rz}
    sizes :: S
    function Partition{Rx, Ry, Rz}() where {Rx, Ry, Rz}
        new{Nothing, Rx, Ry, Rz}(nothing)
    end
    function Partition(sizes::S) where S
        Rx = size(sizes, 1)
        Ry = size(sizes, 2)
        Rz = size(sizes, 3)
        return new{S, Rx, Ry, Rz}(sizes)
    end
end

"""
    Partition(Rx::Number, Ry::Number=1, Rz::Number=1)

Return `Partition` representing the division of a domain into
`Rx` parts in `x` and `Ry` parts in `y` and `Rz` parts in `z`,
where `x, y, z` are the first, second, and third dimension
respectively.
"""
Partition(Rx::Number, Ry::Number=1, Rz::Number=1) = Partition{Rx, Ry, Rz}()
Partition(lengths::Array{Int, 1}) = Partition(reshape(lengths, length(lengths), 1))
Base.size(::Partition{<:Any, Rx, Ry, Rz}) where {Rx, Ry, Rz} = (Rx, Ry, Rz)

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

- `topology` (required): the topology we want the grid to have. It is used to establish connectivity.
                        
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
                     topology = (Periodic, Periodic, Periodic), 
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
        throw(ArgumentError("ranks=($Rx, $Ry, $Rz) [$total_ranks total] inconsistent " *
                            "with number of MPI ranks: $mpi_ranks."))
    end
    
    local_rank         = MPI.Comm_rank(communicator)
    local_index        = rank2index(local_rank, Rx, Ry, Rz)
    local_connectivity = RankConnectivity(local_index, ranks, topology)

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

# RankConnectivity needs to be a tuple since it has to be regularized
# when constructing a grid with a certain topology (topology is not known when
# constructing the architecture)
mutable struct RankConnectivity{E, W, N, S, SW, SE, NW, NE}
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

RankConnectivity(local_index, ranks, ::Nothing) = RankConnectivity(Tuple(nothing for i in 1:8)...)

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

# We want to change the connectivity to adapt to the grid
function regularize_connectivity!(r::RankConnectivity, local_index, ranks, topology)
    r_new = RankConnectivity(local_index, ranks, topology)

    if r_new != r
        @warn "Adapting architecture's connectivity to a $topology grid"
        r.west      = r_new.west     
        r.east      = r_new.east     
        r.south     = r_new.south    
        r.north     = r_new.north    
        r.southwest = r_new.southwest
        r.southeast = r_new.southeast
        r.northwest = r_new.northwest
        r.northeast = r_new.northeast
    end

    return nothing
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
              
