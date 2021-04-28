module Architectures

export AbstractArchitecture, AbstractCPUArchitecture, AbstractGPUArchitecture
export CPU, GPU
export device, architecture, array_type, arch_array

using CUDA
using KernelAbstractions
using Adapt
using OffsetArrays

"""
    AbstractArchitecture

Abstract supertype for architectures supported by Oceananigans.
"""
abstract type AbstractArchitecture end


"""
    AbstractCPUArchitecture

Abstract supertype for CPU architectures supported by Oceananigans.
"""
abstract type AbstractCPUArchitecture <: AbstractArchitecture end

"""
    AbstractGPUArchitecture

Abstract supertype for GPU architectures supported by Oceananigans.
"""
abstract type AbstractGPUArchitecture <: AbstractArchitecture end

"""
    CPU <: AbstractArchitecture

Run Oceananigans on one CPU node. Uses multiple threads if the environment
variable `JULIA_NUM_THREADS` is set.
"""
struct CPU <: AbstractCPUArchitecture end

"""
    GPU <: AbstractArchitecture

Run Oceananigans on a single NVIDIA CUDA GPU.
"""
struct GPU <: AbstractGPUArchitecture end

device(::AbstractCPUArchitecture) = KernelAbstractions.CPU()
device(::AbstractGPUArchitecture) = KernelAbstractions.CUDADevice()

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(::CuArray) = GPU()

array_type(::CPU) = Array
array_type(::GPU) = CuArray

arch_array(::AbstractCPUArchitecture, A::Array) = A
arch_array(::AbstractCPUArchitecture, A::CuArray) = Array(A)
arch_array(::AbstractGPUArchitecture, A::Array) = CuArray(A)
arch_array(::AbstractGPUArchitecture, A::CuArray) = A

const OffsetCPUArray = OffsetArray{FT, N, <:Array} where {FT, N}
const OffsetGPUArray = OffsetArray{FT, N, <:CuArray} where {FT, N}

Adapt.adapt_structure(::CPU, a::OffsetCPUArray) = a
Adapt.adapt_structure(::GPU, a::OffsetGPUArray) = a

Adapt.adapt_structure(::GPU, a::OffsetCPUArray) = OffsetArray(CuArray(a.parent), a.offsets...)
Adapt.adapt_structure(::CPU, a::OffsetGPUArray) = OffsetArray(Array(a.parent), a.offsets...)

end
