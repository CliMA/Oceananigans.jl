module Distributed

export
    MultiArch, child_architecture,
    HaloCommunication, HaloCommunicationBC,
    inject_halo_communication_boundary_conditions,
    DistributedFFTBasedPoissonSolver

using MPI

using Oceananigans.Utils
using Oceananigans.Grids

include("distributed_utils.jl")
include("distributed_grids.jl")
include("multi_architectures.jl")
include("distributed_kernel_launching.jl")
include("halo_communication_bcs.jl")
include("halo_communication.jl")
include("distributed_apply_flux_bcs.jl")
include("distributed_fields.jl")
include("distributed_fft_based_poisson_solver.jl")

end # module
