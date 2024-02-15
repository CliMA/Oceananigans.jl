module Architectures

export AbstractArchitecture
export CPU, GPU, MultiGPU, CUDAGPU
export device, architecture, array_type, arch_array, unified_array, device_copy_to!

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
    CPU <: AbstractArchitecture

An architecture to run Oceananigans on one CPU node. Uses multiple threads
if the environment variable `JULIA_NUM_THREADS` is set.

Example
=======

An instance of CPU architecture.

```jldoctest
julia> using Oceananigans

julia> CPU()
CPU()
```
"""
struct CPU <: AbstractArchitecture end

"""
    GPU{D} <: AbstractArchitecture

An architecture to run Oceananigans on GPU-enabled devices with backend `D`.

Example
=======

An instance of GPU architecture on a CUDA-enabled device.

```jldoctest
julia> using Oceananigans, CUDA

julia> GPU(CUDABackend())
CUDAGPU{CUDABackend}(CUDABackend(false, false))
```
"""
struct GPU{D} <: AbstractArchitecture
    device :: D
end

const CUDAGPU = GPU{<:CUDA.CUDABackend}

GPU() = has_cuda() ? GPU(CUDA.CUDABackend()) : GPU(nothing)

CUDAGPU() = GPU(CUDA.CUDABackend())

#####
##### These methods are extended in DistributedComputations.jl
#####

device(::CPU) = KernelAbstractions.CPU()
device(::CUDAGPU) = CUDA.CUDABackend(; always_inline=true)


architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(::CuArray) = GPU(CUDA.CUDABackend())
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

arch_array(::CPU, a::Array) = a
arch_array(::CPU, a::CuArray) = Array(a)
arch_array(::CUDAGPU, a::Array) = CuArray(a)
arch_array(::CUDAGPU, a::CuArray) = a
arch_array(::CPU, a::BitArray) = a
arch_array(::CUDAGPU, a::BitArray) = CuArray(a)

arch_array(::GPU{D}, a::SubArray{<:Any, <:Any, <:CuArray}) where D = a

arch_array(::CPU, a::SubArray{<:Any, <:Any, <:CuArray}) = Array(a)
arch_array(::CUDAGPU, a::SubArray{<:Any, <:Any, <:Array}) = CuArray(a)

arch_array(::CPU, a::SubArray{<:Any, <:Any, <:Array}) = a

arch_array(::AbstractArchitecture, a::AbstractRange) = a
arch_array(::AbstractArchitecture, ::Nothing) = nothing
arch_array(::AbstractArchitecture, a::Number) = a
arch_array(::AbstractArchitecture, a::Function) = a

arch_array(arch::CPU, a::OffsetArray) = OffsetArray(arch_array(arch, a.parent), a.offsets...)
arch_array(arch::GPU{D}, a::OffsetArray) where D = OffsetArray(arch_array(arch, a.parent), a.offsets...)

cpu_architecture(::CPU) = CPU()
cpu_architecture(::GPU{D}) where D = CPU()

unified_array(::CPU, a) = a
unified_array(::GPU{D}, a) where D = a

function unified_array(::CUDAGPU, arr::AbstractArray) 
    buf = CUDA.Mem.alloc(CUDA.Mem.Unified, sizeof(arr))
    vec = unsafe_wrap(CuArray{eltype(arr),length(size(arr))}, convert(CuPtr{eltype(arr)}, buf), size(arr))
    finalizer(vec) do _
        CUDA.Mem.free(buf)
    end
    copyto!(vec, arr)
    return vec
end

## GPU to GPU copy of contiguous data
@inline function device_copy_to!(dst::CuArray, src::CuArray; async::Bool = false) 
    n = length(src)
    context!(context(src)) do
        GC.@preserve src dst begin
            unsafe_copyto!(pointer(dst, 1), pointer(src, 1), n; async)
        end
    end
    return dst
end

@inline device_copy_to!(dst::Array, src::Array; kw...) = Base.copyto!(dst, src)

@inline unsafe_free!(a::CuArray) = CUDA.unsafe_free!(a)
@inline unsafe_free!(a) = nothing

end # module
