module DistributedComputations

export
    Distributed, Partition, Equal, Fractional,
    child_architecture, reconstruct_global_grid, partition,
    inject_halo_communication_boundary_conditions,
    DistributedFFTBasedPoissonSolver, mpi_initialized, mpi_rank,
    mpi_size, global_barrier, global_communicator,
    @root, @onrank, @distribute, @handshake

using MPI

using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids
using OffsetArrays
using CUDA: CuArray

include("distributed_macros.jl")
include("distributed_architectures.jl")
include("partition_assemble.jl")
include("distributed_grids.jl")
include("distributed_immersed_boundaries.jl")
include("distributed_on_architecture.jl")
include("distributed_kernel_launching.jl")
include("halo_communication_bcs.jl")
include("communication_buffers.jl")
include("distributed_fields.jl")
include("halo_communication.jl")
include("transposable_field.jl")
include("distributed_transpose.jl")
include("plan_distributed_transforms.jl")
include("distributed_fft_based_poisson_solver.jl")
include("distributed_fft_tridiagonal_solver.jl")

end # module
