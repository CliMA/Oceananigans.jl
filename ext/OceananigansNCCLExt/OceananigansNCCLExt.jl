module OceananigansNCCLExt

using NCCL
using CUDA
using MPI
using Oceananigans
using Oceananigans.Utils: launch!
using Oceananigans.DistributedComputations: Distributed

import Oceananigans.DistributedComputations as DC

include("nccl_communicator.jl")
include("nccl_distributed.jl")
include("nccl_transpose.jl")

end # module
