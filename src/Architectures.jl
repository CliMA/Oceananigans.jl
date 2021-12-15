module Architectures

export AbstractArchitecture, AbstractMultiArchitecture
export CPU, GPU
export device, device_event, architecture, array_type, arch_array

using CUDA
using KernelAbstractions
using CUDAKernels
using Adapt
using OffsetArrays

"""
    AbstractArchitecture

Abstract supertype for architectures supported by Oceananigans.
"""
abstract type AbstractArchitecture end

"""
    AbstractMultiArchitecture

Abstract supertype for Distributed architectures supported by Oceananigans.
"""
abstract type AbstractMultiArchitecture <: AbstractArchitecture end

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

## All extension of these methods is done in Distributed.jl

device(::CPU) = KernelAbstractions.CPU()
device(::GPU) = CUDAKernels.CUDADevice()

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(::CuArray) = GPU()

array_type(::CPU) = Array
array_type(::GPU) = CuArray

arch_array(::CPU, A::Array)   = A
arch_array(::CPU, A::CuArray) = Array(A)
arch_array(::GPU, A::Array)   = CuArray(A)
arch_array(::GPU, A::CuArray) = A

const OffsetCPUArray = OffsetArray{FT, N, <:Array} where {FT, N}
const OffsetGPUArray = OffsetArray{FT, N, <:CuArray} where {FT, N}

Adapt.adapt_structure(::CPU, a::OffsetCPUArray) = a
Adapt.adapt_structure(::GPU, a::OffsetGPUArray) = a

Adapt.adapt_structure(::GPU, a::OffsetCPUArray) = OffsetArray(CuArray(a.parent), a.offsets...)
Adapt.adapt_structure(::CPU, a::OffsetGPUArray) = OffsetArray(Array(a.parent), a.offsets...)

device_event(arch) = Event(device(arch))

end
