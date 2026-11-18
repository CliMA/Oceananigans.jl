module OceananigansNCCLExt

using CUDA: CUDA
using MPI: MPI
using NCCL: NCCL
using Oceananigans: Oceananigans
using Oceananigans.Architectures: Architectures as AC, GPU
using Oceananigans.DistributedComputations: DistributedComputations as DC, Distributed

include("nccl_communicator.jl")
include("nccl_distributed.jl")
include("nccl_transpose.jl")

end # module
