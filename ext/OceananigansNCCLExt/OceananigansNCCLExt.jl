module OceananigansNCCLExt

__precompile__(false)  # Required: extension overwrites base methods for NCCL dispatch

using NCCL
using CUDA
using MPI
using Oceananigans
using Oceananigans.Utils: launch!
using Oceananigans.DistributedComputations: Distributed

import Oceananigans.DistributedComputations as DC

include("nccl_communicator.jl")
include("nccl_distributed.jl")
include("nccl_zero_copy_halos.jl")
include("nccl_transpose.jl")
include("nccl_solver.jl")
include("nccl_pipelined_rk3.jl")

end # module
