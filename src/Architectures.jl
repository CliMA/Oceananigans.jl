module Architectures

export AbstractArchitecture, AbstractSerialArchitecture
export CPU, GPU, ReactantState
export device, device!, devices, ndevices, synchronize, architecture, unified_array, device_copy_to!
export array_type, on_architecture, arch_array
export constructors, unpack_constructors, copy_unpack_constructors
export arch_sparse_matrix, child_architecture
using SparseArrays

using KernelAbstractions
using Adapt
using OffsetArrays
const KA = KernelAbstractions

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
    GPU(device)

Return a GPU architecture using `device`.
`device` defauls to CUDA.CUDABackend(always_inline=true)
if CUDA is loaded.
"""
struct GPU{D} <: AbstractSerialArchitecture
    device :: D
end

"""
    ReactantState <: AbstractArchitecture

Run Oceananigans on Reactant.
"""
struct ReactantState <: AbstractSerialArchitecture end

#####
##### These methods are extended in DistributedComputations.jl
#####

device(a::CPU) = KernelAbstractions.CPU()
device(a::GPU) = a.device
devices(a::AbstractArchitecture) = KA.devices(device(a))
device!(a::AbstractArchitecture, i) = KA.device!(device(a), i+1)
ndevices(a::AbstractArchitecture) = KA.ndevices(device(a))
synchronize(a::AbstractArchitecture) = KA.synchronize(device(a))

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(a::SubArray) = architecture(parent(a))
architecture(a::OffsetArray) = architecture(parent(a))
architecture(::SparseMatrixCSC) = CPU()

"""
    child_architecture(arch)

Return `arch`itecture of child processes.
On single-process, non-distributed systems, return `arch`.
"""
child_architecture(arch::AbstractSerialArchitecture) = arch

array_type(::CPU) = Array

# Fallback
on_architecture(arch, a) = a

# Tupled implementation
on_architecture(arch::AbstractSerialArchitecture, t::Tuple) = Tuple(on_architecture(arch, elem) for elem in t)
on_architecture(arch::AbstractSerialArchitecture, nt::NamedTuple) = NamedTuple{keys(nt)}(on_architecture(arch, Tuple(nt)))

# On architecture for array types
on_architecture(::CPU, a::Array) = a
on_architecture(::CPU, a::BitArray) = a
on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:Array}) = a
on_architecture(::CPU, a::StepRangeLen) = a

on_architecture(arch::AbstractSerialArchitecture, a::OffsetArray) =
    OffsetArray(on_architecture(arch, a.parent), a.offsets...)

cpu_architecture(::CPU) = CPU()
cpu_architecture(::GPU) = CPU()
cpu_architecture(::ReactantState) = CPU()

unified_array(::CPU, a) = a
unified_array(::GPU, a) = a

# cu alters the type of `a`, so we convert it back to the correct type
unified_array(::GPU, a::AbstractArray) = map(eltype(a), cu(a; unified = true))

@inline device_copy_to!(dst::Array, src::Array; kw...) = Base.copyto!(dst, src)

@inline unsafe_free!(a)          = nothing

# Convert arguments to GPU-compatible types
@inline convert_to_device(arch, args)  = args
@inline convert_to_device(::CPU, args) = args

# Utils for sparse matrix manipulation
@inline constructors(::CPU, A::SparseMatrixCSC) = (A.m, A.n, A.colptr, A.rowval, A.nzval)
@inline constructors(::CPU, m::Number, n::Number, constr::Tuple) = (m, n, constr...)
@inline constructors(::GPU, m::Number, n::Number, constr::Tuple) = (constr..., (m, n))

@inline unpack_constructors(::GPU, constr::Tuple) = (constr[1], constr[2], constr[3])

@inline copy_unpack_constructors(::GPU, constr::Tuple) = deepcopy((constr[1], constr[2], constr[3]))

@inline arch_sparse_matrix(::CPU, constr::Tuple) = SparseMatrixCSC(constr...)
@inline arch_sparse_matrix(::CPU, A::SparseMatrixCSC)   = A

# Deprecated functions
function arch_array(arch, arr)
    @warn "`arch_array` is deprecated. Use `on_architecture` instead."
    return on_architecture(arch, arr)
end

end # module

