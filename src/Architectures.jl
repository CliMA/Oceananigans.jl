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

Run Oceananigans on one CPU node. Uses multiple threads if the environment
variable `JULIA_NUM_THREADS` is set.
"""
struct CPU <: AbstractSerialArchitecture end

"""
    GPU <: AbstractArchitecture

Run Oceananigans on a single NVIDIA CUDA GPU.
"""
struct GPU <: AbstractSerialArchitecture end

#####
##### These methods are extended in DistributedComputations.jl
#####

device(::CPU) = KernelAbstractions.CPU()
device(::GPU) = CUDA.CUDABackend(; always_inline=true)

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
child_architecture(arch::AbstractSerialArchitecture) = arch

array_type(::CPU) = Array
array_type(::GPU) = CuArray

# Fallback 
on_architecture(arch, a) = a

# Tupled implementation
on_architecture(arch::AbstractSerialArchitecture, t::Tuple) = Tuple(on_architecture(arch, elem) for elem in t)
on_architecture(arch::AbstractSerialArchitecture, nt::NamedTuple) = NamedTuple{keys(nt)}(on_architecture(arch, Tuple(nt)))

# On architecture for array types
on_architecture(::CPU, a::Array) = a
on_architecture(::GPU, a::Array) = CuArray(a)

on_architecture(::CPU, a::CuArray) = Array(a)
on_architecture(::GPU, a::CuArray) = a

on_architecture(::CPU, a::BitArray) = a
on_architecture(::GPU, a::BitArray) = CuArray(a)

on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:CuArray}) = Array(a)
on_architecture(::GPU, a::SubArray{<:Any, <:Any, <:CuArray}) = a

on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:Array}) = a
on_architecture(::GPU, a::SubArray{<:Any, <:Any, <:Array}) = CuArray(a)

on_architecture(arch::AbstractSerialArchitecture, a::OffsetArray) = OffsetArray(on_architecture(arch, a.parent), a.offsets...)

cpu_architecture(::CPU) = CPU()
cpu_architecture(::GPU) = CPU()

unified_array(::CPU, a) = a
unified_array(::GPU, a) = a

# cu alters the type of `a`, so we convert it back to the correct type
unified_array(::GPU, a::AbstractArray) = map(eltype(a), cu(a; unified = true))

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

