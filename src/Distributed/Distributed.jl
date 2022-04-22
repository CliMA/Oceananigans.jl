module Distributed

export
    MultiArch, child_architecture, reconstruct_global_grid, 
    HaloCommunication, HaloCommunicationBC,
    inject_halo_communication_boundary_conditions,
    DistributedFFTBasedPoissonSolver

using MPI

using Oceananigans.Utils
using Oceananigans.Grids

include("distributed_utils.jl")
include("multi_architectures.jl")
include("partition_assemble.jl")
include("distributed_grids.jl")
include("distributed_kernel_launching.jl")
include("halo_communication_bcs.jl")
include("halo_communication.jl")
include("distributed_apply_flux_bcs.jl")
include("distributed_fields.jl")
include("distributed_fft_based_poisson_solver.jl")

end # module
