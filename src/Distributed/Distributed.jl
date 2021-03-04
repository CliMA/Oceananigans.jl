module Distributed

export
    MultiCPU, child_architecture,
    HaloCommunication, HaloCommunicationBC,
    inject_halo_communication_boundary_conditions,
    DistributedFFTBasedPoissonSolver,
    DistributedModel

include("distributed_utils.jl")
include("multi_architectures.jl")
include("halo_communication_bcs.jl")
include("halo_communication.jl")
include("distributed_fft_based_poisson_solver.jl")
include("distributed_solve_for_pressure.jl")
include("distributed_model.jl")

end # module
