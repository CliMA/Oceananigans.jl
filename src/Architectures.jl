module Architectures

export
    AbstractArchitecture, AbstractSerialArchitecture,
    CPU, GPU, ReactantState,
    device, device!, ndevices, synchronize, device_copy_to!,
    array_type, unified_array,
    architecture, child_architecture, on_architecture

using Adapt: Adapt
using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: KernelAbstractions as KA
using OffsetArrays: OffsetArrays, OffsetArray
using SparseArrays: SparseArrays, SparseMatrixCSC

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
`device` defauls to `CUDA.CUDABackend(always_inline=true)`
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

device(a::CPU) = KA.CPU()
device(a::GPU) = a.device
device!(::CPU, i) = nothing
device!(::CPU) = nothing
ndevices(a::CPU) = KA.ndevices(KA.CPU())
ndevices(a::AbstractArchitecture) = KA.ndevices(a.device)
synchronize(a::CPU) = KA.synchronize(KA.CPU())
synchronize(a::AbstractArchitecture) = KA.synchronize(a.device)

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(a::SubArray) = architecture(parent(a))
architecture(a::OffsetArray) = architecture(parent(a))
architecture(::SparseMatrixCSC) = CPU()
architecture(::Type{T}) where {T<:AbstractArray} = architecture(Base.typename(T).wrapper)
architecture(::Type{Array}) = CPU()

# Utils for sparse matrix manipulation
@inline sparse_matrix_constructors(::CPU, A::SparseMatrixCSC) = (A.m, A.n, A.colptr, A.rowval, A.nzval)
@inline sparse_matrix_constructors(::CPU, m::Number, n::Number, constr::Tuple) = (m, n, constr...)
@inline sparse_matrix_constructors(::GPU, m::Number, n::Number, constr::Tuple) = (constr..., (m, n))
@inline sparse_matrix(::CPU, constr::Tuple) = SparseMatrixCSC(constr...)

"""
$(TYPEDSIGNATURES)

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
on_architecture(::CPU, a::StepRangeLen) = a
on_architecture(::CPU, A::SparseMatrixCSC) = A

on_architecture(arch::AbstractSerialArchitecture, a::OffsetArray) =
    OffsetArray(on_architecture(arch, a.parent), a.offsets...)

function on_architecture(arch::AbstractArchitecture, a::SubArray)
    p = on_architecture(arch, parent(a))
    return SubArray(p, parentindices(a))
end

cpu_architecture(::CPU) = CPU()
cpu_architecture(::GPU) = CPU()
cpu_architecture(::ReactantState) = CPU()

Base.summary(::CPU) = "CPU"
Base.summary(gpu::GPU) = "GPU{$(typeof(gpu.device))}"
Base.summary(::ReactantState) = "ReactantState"

unified_array(::CPU, a) = a
unified_array(::GPU, a) = a

@inline device_copy_to!(dst::Array, src::Array; kw...) = Base.copyto!(dst, src)

@inline unsafe_free!(a) = nothing

# CPU kernel arguments are adapted to `CPU()`. Like on GPUs this strips fields down to their data, so
# that kernels do not specialize on field metadata that kernels never use (e.g. boundary conditions:
# without this every distinct combination of boundary-condition types recompiles every kernel).
# Unlike on GPUs, objects that are already CPU-ready are passed through as they are:
#   * grids (see `Grids`): rebuilding them buys nothing on the CPU and only makes the kernel-launching
#     code larger, which can defeat inlining and cause allocations in tight launch loops;
#   * user functions: Adapt recurses into the captured variables of closures, which never terminates
#     for closures that are self-referential through a `Core.Box`.
@inline Adapt.adapt(::CPU, f::Function) = f

# Convert arguments to device-compatible types
@inline convert_to_device(arch, args)  = args
@inline convert_to_device(::CPU, args) = Adapt.adapt(CPU(), args)

end # module
