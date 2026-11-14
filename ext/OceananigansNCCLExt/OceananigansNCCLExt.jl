module OceananigansNCCLExt

using NCCL
using CUDA
using MPI
using Oceananigans
using Oceananigans.DistributedComputations: DistributedComputations as DC, Distributed
using Oceananigans.Utils: launch!

include("nccl_communicator.jl")
include("nccl_distributed.jl")
include("nccl_transpose.jl")

end # module
