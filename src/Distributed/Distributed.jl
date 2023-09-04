module Distributed

export
    DistributedArch, child_architecture, reconstruct_global_grid, 
    inject_halo_communication_boundary_conditions,
    DistributedFFTBasedPoissonSolver

using MPI

using Oceananigans.Utils
using Oceananigans.Grids

include("multi_architectures.jl")
include("partition_assemble.jl")
include("distributed_grids.jl")
include("distributed_kernel_launching.jl")
include("halo_communication_bcs.jl")
include("distributed_fields.jl")
include("halo_communication.jl")
include("interleave_comm_and_comp.jl")
include("parallel_fields.jl")
include("transpose_array.jl")
include("plan_distributed_transforms.jl")
include("distributed_fft_based_poisson2.jl")

end # module
