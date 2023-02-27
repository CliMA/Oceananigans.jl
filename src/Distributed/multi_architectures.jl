using Oceananigans.Architectures
using Oceananigans.Grids: topology, validate_tupled_argument

using CUDA: ndevices, device!

import Oceananigans.Architectures: device, device_event, arch_array, array_type, child_architecture
import Oceananigans.Grids: zeros
import Oceananigans.Fields: using_buffered_communication

struct DistributedArch{A, R, I, ρ, C, γ, B} <: AbstractArchitecture
  child_architecture :: A
          local_rank :: R
         local_index :: I
               ranks :: ρ
        connectivity :: C
        communicator :: γ
end

#####
##### Constructors
#####

"""
    DistributedArch(child_architecture = CPU(); 
                    topology = (Periodic, Periodic, Periodic), 
                    ranks, 
                    use_buffers = false,
                    devices = nothing, 
                    communicator = MPI.COMM_WORLD)

Constructor for a distributed architecture that uses MPI for communications


Positional arguments
=================

- `child_architecture`: Specifies whether the computation is performed on CPUs or GPUs. 
                        Default: `child_architecture = CPU()`.

Keyword arguments
=================

- `topology`: the topology we want the grid to have. It is used to establish connectivity.
              Default: `topology = (Periodic, Periodic, Periodic)`.

- `ranks` (required): A 3-tuple `(Rx, Ry, Rz)` specifying the total processors in the `x`, 
                      `y` and `z` direction. NOTE: support for distributed z direction is 
                      limited, so `Rz = 1` is strongly suggested.

- `use_buffers`: if `true`, buffered halo communication is implemented. If `false`, halos will be 
                 exchanged through views. Buffered communication is not necessary in case of `CPU`
                 execution, but it is necessary for `GPU` execution without CUDA-aware MPI

- `devices`: `GPU` device linked to local rank. The GPU will be assigned based on the 
             local node rank as such `devices[node_rank]`. Make sure to run `--ntasks-per-node` <= `--gres=gpu`.
             If `nothing`, the devices will be assigned automatically based on the available resources

- `communicator`: the MPI communicator, `MPI.COMM_WORLD`. This keyword argument should not be tampered with 
                  if not for testing or developing. Change at your own risk!
"""
function DistributedArch(child_architecture = CPU(); 
                         topology = (Periodic, Periodic, Periodic),
                         ranks,
                         use_buffers = false,
                         devices = nothing, 
                         communicator = MPI.COMM_WORLD)

    MPI.Initialized() || error("Must call MPI.Init() before constructing a MultiCPU.")

    (use_buffers && child_architecture isa CPU) && 
            @warn "Using buffers on CPU architectures is not required (but useful for testing)"

    (!use_buffers && child_architecture isa GPU) && 
            @warn "On GPU architectures not using buffers will lead to a substantial slowdown https://www.open-mpi.org/faq/?category=runcuda#mpi-cuda-support"

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

    B = use_buffers

    return DistributedArch{A, R, I, ρ, C, γ, B}(child_architecture, local_rank, local_index, ranks, local_connectivity, communicator)
end

const ViewsDistributedArch = DistributedArch{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, false}

using_buffered_communication(::DistributedArch{A, R, I, ρ, C, γ, B}) where {A, R, I, ρ, C, γ, B} = B

#####
##### All the architectures
#####

child_architecture(arch::DistributedArch) = arch.child_architecture
device(arch::DistributedArch)             = device(child_architecture(arch))
device_event(arch::DistributedArch)       = device_event(child_architecture(arch))
arch_array(arch::DistributedArch, A)      = arch_array(child_architecture(arch), A)
zeros(FT, arch::DistributedArch, N...)    = zeros(FT, child_architecture(arch), N...)
array_type(arch::DistributedArch)         = array_type(child_architecture(arch))

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

struct RankConnectivity{E, W, N, S, T, B}
      east :: E
      west :: W
     north :: N
     south :: S
       top :: T
    bottom :: B
end

RankConnectivity(; east, west, north, south, top, bottom) =
    RankConnectivity(east, west, north, south, top, bottom)

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

function RankConnectivity(model_index, ranks, topology)
    i, j, k = model_index
    Rx, Ry, Rz = ranks
    TX, TY, TZ = topology

    i_east  = increment_index(i, Rx, TX)
    i_west  = decrement_index(i, Rx, TX)
    j_north = increment_index(j, Ry, TY)
    j_south = decrement_index(j, Ry, TY)
    k_top   = increment_index(k, Rz, TZ)
    k_bot   = decrement_index(k, Rz, TZ)

    r_east  = isnothing(i_east)  ? nothing : index2rank(i_east, j, k, Rx, Ry, Rz)
    r_west  = isnothing(i_west)  ? nothing : index2rank(i_west, j, k, Rx, Ry, Rz)
    r_north = isnothing(j_north) ? nothing : index2rank(i, j_north, k, Rx, Ry, Rz)
    r_south = isnothing(j_south) ? nothing : index2rank(i, j_south, k, Rx, Ry, Rz)
    r_top   = isnothing(k_top)   ? nothing : index2rank(i, j, k_top, Rx, Ry, Rz)
    r_bot   = isnothing(k_bot)   ? nothing : index2rank(i, j, k_bot, Rx, Ry, Rz)

    return RankConnectivity(east=r_east, west=r_west, north=r_north,
                            south=r_south, top=r_top, bottom=r_bot)
end

#####
##### Pretty printing
#####

function Base.show(io::IO, arch::DistributedArch)
    c = arch.connectivity
    print(io, "Distributed architecture (rank $(arch.local_rank)/$(prod(arch.ranks)-1)) [index $(arch.local_index) / $(arch.ranks)]\n",
              "└── child architecture: $(typeof(child_architecture(arch))) \n",
              "└── connectivity:",
              isnothing(c.east) ? "" : " east=$(c.east)",
              isnothing(c.west) ? "" : " west=$(c.west)",
              isnothing(c.north) ? "" : " north=$(c.north)",
              isnothing(c.south) ? "" : " south=$(c.south)",
              isnothing(c.top) ? "" : " top=$(c.top)",
              isnothing(c.bottom) ? "" : " bottom=$(c.bottom)")
end
