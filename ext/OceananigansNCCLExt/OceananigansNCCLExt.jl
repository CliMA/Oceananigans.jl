module OceananigansNCCLExt

using NCCL
using CUDA
using MPI
using Oceananigans
using Oceananigans.Utils: launch!

import Oceananigans.DistributedComputations as DC

include("nccl_communicator.jl")
include("nccl_transpose.jl")
include("nccl_solver.jl")

end # module
