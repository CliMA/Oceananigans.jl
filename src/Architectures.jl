module Architectures

export AbstractArchitecture, AbstractMultiArchitecture
export CPU, CUDAGPU, ROCMGPU
export device, device_event, architecture, array_type, arch_array

using CUDA
using AMDGPU
using KernelAbstractions
using CUDAKernels
using ROCKernels
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
    CUDAGPU <: AbstractArchitecture

Run Oceananigans on a single NVIDIA CUDA GPU.
"""
struct CUDAGPU <: AbstractArchitecture end

"""
    ROCMGPU <: AbstractArchitecture

Run Oceananigans on a single AMD ROCM GPU.
"""
struct ROCMGPU <: AbstractArchitecture end

#####
##### These methods are extended in Distributed.jl
#####

device(::CPU) = KernelAbstractions.CPU()
device(::CUDAGPU) = CUDAKernels.CUDADevice()
device(::ROCMGPU) = ROCKernels.ROCDevice()

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(::CuArray) = CUDAGPU()
architecture(::ROCArray) = ROCMGPU()
architecture(a::SubArray) = architecture(parent(a))
architecture(a::OffsetArray) = architecture(parent(a))

"""
    child_architecture(arch)

Return `arch`itecture of child processes.
On single-process, non-distributed systems, return `arch`.
"""
child_architecture(arch) = arch

array_type(::CPU) = Array
array_type(::CUDAGPU) = CuArray
array_type(::ROCMGPU) = ROCArray

arch_array(::CPU, a::Array)   = a
arch_array(::CPU, a::CuArray) = Array(a)
arch_array(::CPU, a::ROCArray) = Array(a)
arch_array(::CUDAGPU, a::Array)   = CuArray(a)
arch_array(::CUDAGPU, a::CuArray) = a
arch_array(::ROCMGPU, a::Array) = ROCArray(a)
arch_array(::ROCMGPU, a::ROCArray) = a

arch_array(arch, a::AbstractRange) = a
arch_array(arch, a::OffsetArray) = OffsetArray(arch_array(arch, a.parent), a.offsets...)
arch_array(arch, ::Nothing) = nothing
arch_array(arch, a::Number) = a

device_event(arch) = Event(device(arch))

end
