module Architectures

export AbstractArchitecture
export CPU, GPU, MultiGPU
export device, architecture, array_type, arch_array, unified_array, device_copy_to!

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

#####
##### These methods are extended in Distributed.jl and CUDAExt.jl
#####

device(::CPU) = KernelAbstractions.CPU()

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(a::SubArray) = architecture(parent(a))
architecture(a::OffsetArray) = architecture(parent(a))

available(::CPU) = true

"""
    child_architecture(arch)

Return `arch`itecture of child processes.
On single-process, non-distributed systems, return `arch`.
"""
child_architecture(arch) = arch

array_type(::CPU) = Array

arch_array(::CPU, a::Array) = a
arch_array(::CPU, a::SubArray{<:Any, <:Any, <:Array}) = a
arch_array(arch, a::AbstractRange) = a
arch_array(arch, a::OffsetArray) = OffsetArray(arch_array(arch, a.parent), a.offsets...)
arch_array(arch, ::Nothing)   = nothing
arch_array(arch, a::Number)   = a
arch_array(arch, a::Function) = a

unified_array(::CPU, a) = a
 
@inline device_copy_to!(dst::Array, src::Array; kw...) = Base.copyto!(dst, src)

@inline unsafe_free!(a)          = nothing

end # module
