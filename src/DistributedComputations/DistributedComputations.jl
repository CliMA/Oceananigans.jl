module DistributedComputations

export
    Distributed, Partition, Equal, Fractional, 
    child_architecture, reconstruct_global_grid, partition,
    inject_halo_communication_boundary_conditions,
    DistributedFFTBasedPoissonSolver

using MPI

using Oceananigans.Utils
using Oceananigans.Grids

include("distributed_architectures.jl")
include("partition_assemble.jl")
include("distributed_grids.jl")
include("distributed_immersed_boundaries.jl")
include("distributed_on_architecture.jl")
include("distributed_kernel_launching.jl")
include("halo_communication_bcs.jl")
include("distributed_fields.jl")
include("halo_communication.jl")
include("transposable_field.jl")
include("distributed_transpose.jl")
include("plan_distributed_transforms.jl")
include("distributed_fft_based_poisson_solver.jl")
include("distributed_fft_tridiagonal_solver.jl")

end # module
