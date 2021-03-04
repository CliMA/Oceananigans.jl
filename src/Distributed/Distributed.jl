module Distributed

export
    MultiCPU,
    HaloCommunication, HaloCommunicationBC,
    DistributedFFTBasedPoissonSolver,
    DistributedModel

include("distributed_utils.jl")
include("multi_architectures.jl")
include("halo_communication_bcs.jl")
include("halo_communication.jl")
include("distributed_fft_based_poisson_solver.jl")
include("distributed_model.jl")

end # module
