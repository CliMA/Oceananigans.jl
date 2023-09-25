module Architectures

export AbstractArchitecture
export CPU, GPU, MultiGPU, MetalBackend
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

Run Oceananigans on a single GPU `device`.
"""
struct GPU{D} <: AbstractArchitecture 
    device :: D
end

function GPU()
    if CUDA.has_cuda()
        return GPU(CUDA.CUDABackend(; always_inline = true))
    elseif has_metal_device()
        return GPU(Metal.MetalBackend())
    else
        error("No compatible GPU detected")
    end
end

"""
    has_metal_device()

Returns true if a metal device is present (not a current function available from Metal.jl)
"""
function has_metal_device()
    try 
        Metal.current_device()
        return true
    catch
        return false
    end
end

#####
##### These methods are extended in DistributedComputations.jl
#####

device(::CPU) = KernelAbstractions.CPU()
device(::GPU{<:CUDABackend}) = CUDA.CUDABackend(; always_inline=true)
device(::GPU{<:MetalBackend}) = Metal.MetalBackend()

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(::CuArray) = GPU(CUDA.CUDABackend(; always_inline=true))
architecture(::MtlArray) = GPU(Metal.MetalBackend())
architecture(a::SubArray) = architecture(parent(a))
architecture(a::OffsetArray) = architecture(parent(a))

"""
    child_architecture(arch)

Return `arch`itecture of child processes.
On single-process, non-distributed systems, return `arch`.
"""
child_architecture(arch) = arch

array_type(::CPU) = Array
array_type(::GPU{<:CUDABackend}) = CuArray
archt_type(::GPU{<:MetalBackend}) = MtlArray

arch_array(::CPU, a::Array)   = a
arch_array(::CPU, a) = Array(a)
arch_array(::GPU{<:CUDABackend}, a::CuArray) = a
arch_array(::GPU{<:CUDABackend}, a)   = CuArray(a)
arch_array(::GPU{<:MetalBackend}, a::MtlArray) = a
arch_array(::GPU{<:MetalBackend}, a) = MtlArray(a)

arch_array(::GPU{<:CUDABackend}, a::SubArray{<:Any, <:Any, <:CuArray}) = a
arch_array(::CPU, a::SubArray{<:Any, <:Any, <:CuArray}) = Array(a)
arch_array(::GPU{<:MetalBackend}, a::SubArray{<:Any, <:Any, <:CuArray}) = MtlArray(a)

arch_array(::GPU{<:CUDABackend}, a::SubArray{<:Any, <:Any, <:Array}) = CuArray(a)
arch_array(::CPU, a::SubArray{<:Any, <:Any, <:Array}) = a
arch_array(::GPU{<:MetalBackend}, a::SubArray{<:Any, <:Any, <:Array}) = MtlArray(a)

arch_array(::GPU{<:CUDABackend}, a::SubArray{<:Any, <:Any, <:MtlArray}) = CuArray(a)
arch_array(::CPU, a::SubArray{<:Any, <:Any, <:MtlArray}) = Array(a)
arch_array(::GPU{<:MetalBackend}, a::SubArray{<:Any, <:Any, <:MtlArray}) = a

arch_array(::CPU, a::AbstractRange) = a
arch_array(::CPU, ::Nothing)   = nothing
arch_array(::CPU, a::Number)   = a
arch_array(::CPU, a::Function) = a

arch_array(::GPU{<:CUDABackend}, a::AbstractRange) = a
arch_array(::GPU{<:CUDABackend}, ::Nothing)   = nothing
arch_array(::GPU{<:CUDABackend}, a::Number)   = a
arch_array(::GPU{<:CUDABackend}, a::Function) = a

# not sure why we can't just have arch_array(<:Any, a::...) = a
arch_array(::GPU{<:MetalBackend}, a::AbstractRange) = a
arch_array(::GPU{<:MetalBackend}, ::Nothing)   = nothing
arch_array(::GPU{<:MetalBackend}, a::Number)   = a
arch_array(::GPU{<:MetalBackend}, a::Function) = a

arch_array(arch::CPU, a::OffsetArray) = OffsetArray(arch_array(arch, a.parent), a.offsets...)
arch_array(arch::GPU{<:CUDABackend}, a::OffsetArray) = OffsetArray(arch_array(arch, a.parent), a.offsets...)
arch_array(arch::GPU{<:MetalBackend}, a::OffsetArray) = OffsetArray(arch_array(arch, a.parent), a.offsets...)

unified_array(::CPU, a) = a
unified_array(::GPU{<:CUDABackend}, a) = a
unified_array(::GPU{<:MetalBackend}, a) = a

# not sure what todo with MetalBackends for this
function unified_array(::GPU{<:CUDABackend}, arr::AbstractArray) 
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
