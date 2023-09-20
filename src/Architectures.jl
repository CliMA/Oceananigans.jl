module Architectures

export AbstractArchitecture
export CPU, GPU, MultiGPU
export device, architecture, array_type, arch_array, unified_array, device_copy_to!

using CUDA
using Metal
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
    MetalBackend <: AbstractArchitecture

Run Oceananigans on a single M1 GPU.
"""
struct MetalBackend <: AbstractArchitecture end

#####
##### These methods are extended in DistributedComputations.jl
#####

device(::CPU) = KernelAbstractions.CPU()
device(::GPU) = CUDA.CUDABackend(; always_inline=true)
device(::MetalBackend) = Metal.MetalBackend()

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(::CuArray) = GPU()
architecture(::MetalBackend) = MetalBackend()
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
archt_type(::MetalBackend) = MtlArray

arch_array(::CPU, a::Array)   = a
arch_array(::CPU, a) = Array(a)
arch_array(::GPU, a::CuArray) = a
arch_array(::GPU, a)   = CuArray(a)
arch_array(::MetalBackend, a::MtlArray) = a
arch_array(::MetalBackend, a) = MtlArray(a)

arch_array(::GPU, a::SubArray{<:Any, <:Any, <:CuArray}) = a
arch_array(::CPU, a::SubArray{<:Any, <:Any, <:CuArray}) = Array(a)
arch_array(::MetalBackend, a::SubArray{<:Any, <:Any, <:CuArray}) = MtlArray(a)

arch_array(::GPU, a::SubArray{<:Any, <:Any, <:Array}) = CuArray(a)
arch_array(::CPU, a::SubArray{<:Any, <:Any, <:Array}) = a
arch_array(::MetalBackend, a::SubArray{<:Any, <:Any, <:Array}) = MtlArray(a)

arch_array(::GPU, a::SubArray{<:Any, <:Any, <:MtlArray}) = CuArray(a)
arch_array(::CPU, a::SubArray{<:Any, <:Any, <:MtlArray}) = Array(a)
arch_array(::MetalBackend, a::SubArray{<:Any, <:Any, <:MtlArray}) = a

arch_array(::CPU, a::AbstractRange) = a
arch_array(::CPU, ::Nothing)   = nothing
arch_array(::CPU, a::Number)   = a
arch_array(::CPU, a::Function) = a

arch_array(::GPU, a::AbstractRange) = a
arch_array(::GPU, ::Nothing)   = nothing
arch_array(::GPU, a::Number)   = a
arch_array(::GPU, a::Function) = a

# not sure why we can't just have arch_array(<:Any, a::...) = a
arch_array(::MetalBackend, a::AbstractRange) = a
arch_array(::MetalBackend, ::Nothing)   = nothing
arch_array(::MetalBackend, a::Number)   = a
arch_array(::MetalBackend, a::Function) = a

arch_array(arch, a::OffsetArray) = OffsetArray(arch_array(arch, a.parent), a.offsets...)

unified_array(::CPU, a) = a
unified_array(::GPU, a) = a
unified_array(::MetalBackend, a) = a

# not sure what todo with MetalBackends for this
function unified_array(::GPU, arr::AbstractArray) 
    buf = Mem.alloc(Mem.Unified, sizeof(arr))
    vec = unsafe_wrap(CuArray{eltype(arr),length(size(arr))}, convert(CuPtr{eltype(arr)}, buf), size(arr))
    finalizer(vec) do _
        Mem.free(buf)
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
@inline unsafe_free!(a)          = nothing

end # module
