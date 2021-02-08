module Architectures

export
    @hascuda,
    AbstractArchitecture, CPU, GPU,
    device, architecture, array_type, arch_array

using CUDA

using KernelAbstractions

"""
    AbstractArchitecture

Abstract supertype for architectures supported by Oceananigans.
"""
abstract type AbstractArchitecture end

"""
    CPU <: AbstractArchitecture

Run Oceananigans on one CPU node. Uses multiple threads if the environment
variable `JULIA_NUM_THREADS` is set.
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

device(::CPU) = KernelAbstractions.CPU()
device(::GPU) = KernelAbstractions.CUDADevice()

architecture(::Number)  = nothing
architecture(::Array)   = CPU()
architecture(::CuArray) = GPU()

array_type(::CPU) = Array
array_type(::GPU) = CuArray

arch_array(::CPU, A::Array) = A
arch_array(::CPU, A::CuArray) = Array(A)
arch_array(::GPU, A::Array) = CuArray(A)
arch_array(::GPU, A::CuArray) = A

end
