module OceananigansNCCLExt

using NCCL
using CUDA
using MPI
using Oceananigans
using Oceananigans.Utils: launch!
using Oceananigans.DistributedComputations: DistributedComputations as DC, Distributed

include("nccl_communicator.jl")
include("nccl_distributed.jl")
include("nccl_transpose.jl")

end # module
