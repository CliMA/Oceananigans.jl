module Architectures

export AbstractArchitecture, AbstractMultiArchitecture
export CPU, GPU, MultiGPU
export device, device_event, architecture, array_type, arch_array

using CUDA
using KernelAbstractions
using CUDAKernels
using Adapt
using OffsetArrays

# Adapt CUDAKernels to multiple devices by splitting stream pool
import CUDAKernels: next_stream

if CUDA.has_cuda_gpu()     
    const FREE_STREAMS = CUDA.CuStream[]
    const STREAMS = CUDA.CuStream[]
    const STREAM_GC_THRESHOLD = Ref{Int}(16)

    # This code is loaded after an `@init` step
    if haskey(ENV, "KERNELABSTRACTIONS_STREAMS_GC_THRESHOLD")
        global STREAM_GC_THRESHOLD[] = parse(Int, ENV["KERNELABSTRACTIONS_STREAMS_GC_THRESHOLD"])
    end

    const STREAM_GC_LOCK = Threads.ReentrantLock()

    const FREE_STREAMS_D = Dict{CUDA.CuContext,Array{CUDA.CuStream,1}}()
    const STREAMS_D      = Dict{CUDA.CuContext,Array{CUDA.CuStream,1}}()
    function next_stream()
        ctx = CUDA.current_context()
        lock(STREAM_GC_LOCK) do
            # see if there is a compatible free stream
            FREE_STREAMS_CT  = get!(FREE_STREAMS_D, ctx) do
            CUDA.CuStream[]
            end
            if !isempty(FREE_STREAMS_CT)
            return pop!(FREE_STREAMS_CT)
            end

            # GC to recover streams that are not busy
            STREAMS_CT  = get!(STREAMS_D, ctx) do
                CUDA.CuStream[]
            end
            if length(STREAMS_CT) > STREAM_GC_THRESHOLD[]
                for stream in STREAMS_CT
                    if CUDA.query(stream)
                        push!(FREE_STREAMS_CT, stream)
                    end
                end
            end

            # if there is a compatible free stream after GC, return that stream
            if !isempty(FREE_STREAMS_CT)
                return pop!(FREE_STREAMS_CT)
            end

            # no compatible free stream available so create a new one
            stream = CUDA.CuStream(flags = CUDA.STREAM_NON_BLOCKING)
            push!(STREAMS_CT, stream)
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

"""
    MultiGPU <: AbstractArchitecture

Run Oceananigans on multiple NVIDIA CUDA GPUs connected to the same host.
Can be used only in connection with `MultiRegionGrid`
"""
struct MultiGPU <: AbstractArchitecture end

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
arch_array(arch, ::Nothing) = nothing
arch_array(arch, a::Number) = a

device_event(arch) = Event(device(arch))

end
