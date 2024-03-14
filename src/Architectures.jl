module Architectures

export AbstractArchitecture, AbstractSerialArchitecture
export CPU, GPU
export device, architecture, unified_array, device_copy_to!
export array_type, on_architecture, arch_array

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
    AbstractSerialArchitecture

Abstract supertype for serial architectures supported by Oceananigans.
"""
abstract type AbstractSerialArchitecture <: AbstractArchitecture end

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
struct CPU <: AbstractSerialArchitecture end

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
struct GPU{D} <: AbstractSerialArchitecture 
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

# Fallback 
on_architecture(arch, a) = a

# Tupled implementation
on_architecture(arch::AbstractSerialArchitecture, t::Tuple) = Tuple(on_architecture(arch, elem) for elem in t)
on_architecture(arch::AbstractSerialArchitecture, nt::NamedTuple) = NamedTuple{keys(nt)}(on_architecture(arch, Tuple(nt)))

# On architecture for array types
on_architecture(::CPU,     a::Array) = a
on_architecture(::CUDAGPU, a::Array) = CuArray(a)

on_architecture(::CPU,     a::CuArray) = Array(a)
on_architecture(::CUDAGPU, a::CuArray) = a

on_architecture(::CPU,     a::BitArray) = a
on_architecture(::CUDAGPU, a::BitArray) = CuArray(a)

on_architecture(::CPU,     a::SubArray{<:Any, <:Any, <:CuArray}) = Array(a)
on_architecture(::CUDAGPU, a::SubArray{<:Any, <:Any, <:CuArray}) = a

on_architecture(::CPU,     a::SubArray{<:Any, <:Any, <:Array}) = a
on_architecture(::CUDAGPU, a::SubArray{<:Any, <:Any, <:Array}) = CuArray(a)

on_architecture(arch::AbstractSerialArchitecture, a::OffsetArray) = OffsetArray(on_architecture(arch, a.parent), a.offsets...)

cpu_architecture(::CPU) = CPU()
cpu_architecture(::GPU) = CPU()

unified_array(::CPU, a) = a
unified_array(::GPU, a) = a

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

# Convert arguments to GPU-compatible types
@inline convert_args(::CPU, args) = args
@inline convert_args(::GPU, args) = CUDA.cudaconvert(args)
@inline convert_args(::GPU, args::Tuple) = map(CUDA.cudaconvert, args)

# Deprecated functions
function arch_array(arch, arr) 
    @warn "`arch_array` is deprecated. Use `on_architecture` instead."
    return on_architecture(arch, arr)
end

end # module

