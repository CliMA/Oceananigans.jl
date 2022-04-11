module Architectures

export AbstractArchitecture, AbstractMultiArchitecture
export CPU, GPU, MultiGPU
export device, device_event, architecture, array_type, arch_array, unified_array

using CUDA
using KernelAbstractions
using CUDAKernels
using Adapt
using OffsetArrays

# Adapt CUDAKernels to multiple devices by splitting stream pool
import CUDAKernels: next_stream

if CUDA.has_cuda_gpu()     
using CUDAKernels: STREAM_GC_LOCK

    DEVICE_FREE_STREAMS = Tuple(CUDA.CuStream[] for dev in 1:length(CUDA.devices()))
    DEVICE_STREAMS      = Tuple(CUDA.CuStream[] for dev in 1:length(CUDA.devices()))
    const DEVICE_STREAM_GC_THRESHOLD = Ref{Int}(16)

    function next_stream()
        lock(STREAM_GC_LOCK) do
            handle = CUDA.device().handle + 1
            if !isempty(DEVICE_FREE_STREAMS[handle])
                return pop!(DEVICE_FREE_STREAMS[handle])
            end

            if length(DEVICE_STREAMS[handle]) > DEVICE_STREAM_GC_THRESHOLD[]
                for stream in DEVICE_STREAMS[handle]
                    if CUDA.query(stream)
                        push!(DEVICE_FREE_STREAMS[handle], stream)
                    end
                end
            end

            if !isempty(DEVICE_FREE_STREAMS[handle])
                return pop!(DEVICE_FREE_STREAMS[handle])
            end
            stream = CUDA.CuStream(flags = CUDA.STREAM_NON_BLOCKING)
            push!(DEVICE_STREAMS[handle], stream)
            return stream
        end
    end
end

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

#####
##### These methods are extended in Distributed.jl
#####

device(::CPU) = KernelAbstractions.CPU()
device(::GPU) = CUDAKernels.CUDADevice()

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(::CuArray) = GPU()
architecture(a::SubArray) = architecture(parent(a))
architecture(a::OffsetArray) = architecture(parent(a))

"""
    child_architecture(arch)

Return `arch`itecture of child processes.
On single-process, non-distributed systems, return `arch`.
"""
child_architecture(arch) = arch

array_type(::CPU) = Array
array_type(::GPU) = CuArray

arch_array(::CPU, a::Array)   = a
arch_array(::CPU, a::CuArray) = Array(a)
arch_array(::GPU, a::Array)   = CuArray(a)
arch_array(::GPU, a::CuArray) = a

arch_array(arch, a::AbstractRange) = a
arch_array(arch, a::OffsetArray) = OffsetArray(arch_array(arch, a.parent), a.offsets...)
arch_array(arch, ::Nothing)   = nothing
arch_array(arch, a::Number)   = a
arch_array(arch, a::Function) = a

unified_array(::CPU, a) = a
unified_array(::GPU, a) = a

function unified_array(::GPU, arr::AbstractArray) 
    buf = Mem.alloc(Mem.Unified, sizeof(arr))
    vec = unsafe_wrap(CuArray{eltype(arr),length(size(arr))}, convert(CuPtr{eltype(arr)}, buf), size(arr))
    finalizer(vec) do _
        Mem.free(buf)
    end
    copyto!(vec, arr)
    return vec
end

device_event(arch) = Event(device(arch))

end
