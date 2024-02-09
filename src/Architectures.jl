module Architectures

export AbstractArchitecture
export CPU, GPU, MultiGPU, CUDAGPU, ROCmGPU
export device, architecture, array_type, arch_array, unified_array, device_copy_to!

using CUDA
using AMDGPU
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
    GPU{D} <: AbstractArchitecture

Run Oceananigans on a single Nvidia CUDA or AMD ROCm GPU.
"""
struct GPU{D} <: AbstractArchitecture
    device :: D
end

const CUDAGPU = GPU{<:CUDA.CUDABackend}
const ROCmGPU = GPU{<:AMDGPU.ROCBackend}

GPU() = has_cuda() ? GPU(CUDA.CUDABackend()) : GPU(AMDGPU.ROCBackend())
CUDAGPU() = GPU(CUDA.CUDABackend())
ROCmGPU() = GPU(AMDGPU.ROCBackend())

#####
##### These methods are extended in DistributedComputations.jl
#####

device(::CPU) = KernelAbstractions.CPU()
device(::CUDAGPU) = CUDA.CUDABackend(;always_inline=true)
device(::ROCmGPU) = AMDGPU.ROCBackend()

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(::CuArray) = GPU(CUDA.CUDABackend())
architecture(::ROCArray) = GPU(AMDGPU.ROCBackend())
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
array_type(::ROCmGPU) = ROCArray

arch_array(::CPU, a::Array)   = a
arch_array(::CPU, a::CuArray) = Array(a)
arch_array(::CPU, a::ROCArray) = Array(a)
arch_array(::CUDAGPU, a::Array) = CuArray(a)
arch_array(::ROCmGPU, a::Array) = ROCArray(a)
arch_array(::CUDAGPU, a::CuArray) = a
arch_array(::ROCmGPU, a::ROCArray) = a
arch_array(::CPU, a::BitArray) = a
arch_array(::CUDAGPU, a::BitArray) = CuArray(a)
arch_array(::ROCmGPU, a::BitArray) = ROCArray(a)

arch_array(::CUDAGPU, a::SubArray{<:Any, <:Any, <:CuArray}) = a
arch_array(::ROCmGPU, a::SubArray{<:Any, <:Any, <:ROCArray}) = a

arch_array(::CPU, a::SubArray{<:Any, <:Any, <:CuArray}) = Array(a)
arch_array(::CUDAGPU, a::SubArray{<:Any, <:Any, <:Array}) = CuArray(a)
arch_array(::ROCmGPU, a::SubArray{<:Any, <:Any, <:Array}) = ROCArray(a)

arch_array(::CPU, a::SubArray{<:Any, <:Any, <:Array}) = a

arch_array(::CPU, a::AbstractRange) = a
arch_array(::CPU, ::Nothing)   = nothing
arch_array(::CPU, a::Number)   = a
arch_array(::CPU, a::Function) = a

arch_array(::CUDAGPU, a::AbstractRange) = a
arch_array(::CUDAGPU, ::Nothing)   = nothing
arch_array(::CUDAGPU, a::Number)   = a
arch_array(::CUDAGPU, a::Function) = a

arch_array(::ROCmGPU, a::AbstractRange) = a
arch_array(::ROCmGPU, ::Nothing)   = nothing
arch_array(::ROCmGPU, a::Number)   = a
arch_array(::ROCmGPU, a::Function) = a

arch_array(arch::CPU, a::OffsetArray) = OffsetArray(arch_array(arch, a.parent), a.offsets...)
arch_array(arch::CUDAGPU, a::OffsetArray) = OffsetArray(arch_array(arch, a.parent), a.offsets...)
arch_array(arch::ROCmGPU, a::OffsetArray) = OffsetArray(arch_array(arch, a.parent), a.offsets...)

cpu_architecture(::CPU) = CPU()
cpu_architecture(::CUDAGPU) = CPU()
cpu_architecture(::ROCmGPU) = CPU()

unified_array(::CPU, a) = a
unified_array(::CUDAGPU, a) = a
unified_array(::ROCmGPU, a) = a

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

@inline function device_copy_to!(dst::ROCArray, src::ROCArray; async::Bool = false) 
    AMDGPU.mem.transfer!(dst.buf, src.buf, sizeof(src))
    return dst
end
 
@inline device_copy_to!(dst::Array, src::Array; kw...) = Base.copyto!(dst, src)

@inline unsafe_free!(a::CuArray)  = CUDA.unsafe_free!(a)
@inline unsafe_free!(a::ROCArray) = AMDGPU.unsafe_free!(a)
@inline unsafe_free!(a)           = nothing

end # module
