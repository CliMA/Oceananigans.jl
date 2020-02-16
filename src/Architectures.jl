module Architectures

import GPUifyLoops
using CUDAapi: has_cuda

export
    @hascuda,
    AbstractArchitecture, CPU, GPU,
    device, architecture, array_type

"""
    AbstractArchitecture

Abstract supertype for architectures supported by Oceananigans.
"""
abstract type AbstractArchitecture end

"""
    CPU <: AbstractArchitecture

Run Oceananigans on a single-core of a CPU.
"""
struct CPU <: AbstractArchitecture end

"""
    GPU <: AbstractArchitecture

Run Oceananigans on a single NVIDIA CUDA GPU.
"""
struct GPU <: AbstractArchitecture end

"""
    @hascuda expr

A macro to compile and execute `expr` only if CUDA is installed and available. Generally used to
wrap expressions that can only be compiled if `CuArrays` and `CUDAnative` can be loaded.
"""
macro hascuda(expr)
    return has_cuda() ? :($(esc(expr))) : :(nothing)
end

device(::CPU) = GPUifyLoops.CPU()
device(::GPU) = GPUifyLoops.CUDA()

@hascuda using CuArrays

         architecture(::Array)   = CPU()
@hascuda architecture(::CuArray) = GPU()

         array_type(::CPU) = Array
@hascuda array_type(::GPU) = CuArray

end
