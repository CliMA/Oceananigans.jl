module OceananigansCUDAExt

using InteractiveUtils: versioninfo
using CUDA: CUDA, CuArray, CuContext, CuDevice, CuDeviceArray, CuPtr, context,
    context!, cu, CUDA.CUDABackend
# TODO: when we'll support CUDA.jl v6 only, add `cuSPARSE`/`CUDACore`/`cuFFT` to
# list of triggers of this extensions, and simplify the conditions below.
if isdefined(CUDA, :cuSPARSE)
    using CUDA.cuSPARSE: CuSparseMatrixCSC
else
    using CUDA.CUSPARSE: CuSparseMatrixCSC
end
if isdefined(CUDA, :cuFFT)
    using CUDA.cuFFT: cuFFT
else
    const cuFFT = CUDA.CUFFT
end
using Adapt: Adapt
using OffsetArrays: OffsetArray
using GPUArraysCore: allowscalar
using Oceananigans.Utils: linear_expand, __linear_ndrange, MappedCompilerMetadata

import Oceananigans.Architectures as AC
import Oceananigans.BoundaryConditions as BC
import Oceananigans.DistributedComputations as DC
import Oceananigans.Fields as FD
import Oceananigans.MultiRegion as MR
import Oceananigans.Utils: apply_regionally!
import Oceananigans.Grids as GD
import Oceananigans.Solvers as SO
import Oceananigans.Utils as UT
import SparseArrays: SparseMatrixCSC
import KernelAbstractions: __iterspace, __dynamic_checkbounds, __validindex
import KernelAbstractions as KA
import Oceananigans.DistributedComputations: Distributed

const GPUVar = Union{CuArray, CuContext, CuPtr, Ptr}

function __init__()
    if CUDA.functional()
        @debug "CUDA-enabled GPU(s) detected:"
        for (gpu, dev) in enumerate(CUDA.devices())
            @debug "$dev: $(CUDA.name(dev))"
        end

        allowscalar(false)
    end

    # Cached launch of configured kernels (below): `OCEANANIGANS_CACHED_KERNEL_LAUNCH=false` takes the standard
    # launch everywhere; `OCEANANIGANS_CHECK_CACHED_KERNEL_LAUNCH=true` asserts on every cached launch that
    # re-converting its arguments leaves them bit-identical.
    CACHED_KERNEL_LAUNCH[] = lowercase(get(ENV, "OCEANANIGANS_CACHED_KERNEL_LAUNCH", "true")) != "false"
    CHECK_CACHED_KERNEL_LAUNCH[] = lowercase(get(ENV, "OCEANANIGANS_CHECK_CACHED_KERNEL_LAUNCH", "false")) == "true"
end

const CUDAGPU = AC.GPU{<:CUDABackend}
CUDAGPU() = AC.GPU(CUDABackend(always_inline=true))

# Keep default CUDA backend
function AC.GPU()
    if CUDA.has_cuda_gpu()
        return CUDAGPU()
    else
        msg = """We cannot make a GPU with the CUDA backend:
                 a CUDA GPU was not found!"""
        throw(ArgumentError(msg))
    end
end

function UT.versioninfo_with_gpu(::CUDAGPU)
    s = sprint(versioninfo)
    gpu_name = CUDA.CuDevice(0) |> CUDA.name
    return "CUDA GPU: $gpu_name"
end

Base.summary(::CUDAGPU) = "CUDAGPU"
AC.device!(::CUDAGPU, i) = CUDA.device!(i)

AC.architecture(::CuArray) = CUDAGPU()
AC.architecture(::Type{CuArray}) = CUDAGPU()
AC.architecture(::CuDeviceArray) = CUDAGPU()
AC.architecture(::Type{CuDeviceArray}) = CUDAGPU()
AC.architecture(::CuSparseMatrixCSC) = CUDAGPU()
AC.array_type(::AC.GPU{CUDABackend}) = CuArray

AC.on_architecture(::CUDAGPU, a::Number) = a
AC.on_architecture(::AC.CPU, a::CuArray) = Array(a)
AC.on_architecture(::CUDAGPU, a::Array) = CuArray(a)
AC.on_architecture(::CUDAGPU, a::CuArray) = a
AC.on_architecture(::CUDAGPU, a::BitArray) = CuArray(a)
AC.on_architecture(::CUDAGPU, a::StepRangeLen) = a
AC.on_architecture(arch::Distributed, a::CuArray) = AC.on_architecture(AC.child_architecture(arch), a)

@inline AC.sparse_matrix_constructors(::AC.GPU{CUDABackend}, A::SparseMatrixCSC) = (CuArray(A.colptr), CuArray(A.rowval), CuArray(A.nzval),  (A.m, A.n))
@inline AC.sparse_matrix_constructors(::AC.CPU, A::CuSparseMatrixCSC) = (A.dims[1], A.dims[2], Int64.(Array(A.colPtr)), Int64.(Array(A.rowVal)), Array(A.nzVal))
@inline AC.sparse_matrix_constructors(::AC.GPU{CUDABackend}, A::CuSparseMatrixCSC) = (A.colPtr, A.rowVal, A.nzVal,  A.dims)

@inline AC.sparse_matrix(::AC.GPU{CUDABackend}, constr::Tuple) = CuSparseMatrixCSC(constr...)

@inline AC.on_architecture(::AC.CPU, A::CuSparseMatrixCSC)              = SparseMatrixCSC(AC.sparse_matrix_constructors(AC.CPU(), A)...)
@inline AC.on_architecture(::AC.GPU{CUDABackend}, A::SparseMatrixCSC)   = CuSparseMatrixCSC(AC.sparse_matrix_constructors(AC.GPU(), A)...)
@inline AC.on_architecture(::AC.GPU{CUDABackend}, A::CuSparseMatrixCSC) = A

# cu alters the type of `a`, so we convert it back to the correct type
AC.unified_array(::CUDAGPU, a::AbstractArray) = map(eltype(a), cu(a; unified = true))

## GPU to GPU copy of contiguous data
@inline function AC.device_copy_to!(dst::CuArray, src::CuArray; async::Bool = false)
    n = length(src)
    context!(context(src)) do
        GC.@preserve src dst begin
            unsafe_copyto!(pointer(dst, 1), pointer(src, 1), n; async)
        end
    end
    return dst
end

#####
##### 32-bit index arithmetic for kernel arguments
#####
##### On the device a `Field` is an `OffsetArray` over a `CuDeviceArray`, whose index arithmetic is 64-bit: every
##### `c[i, j, k]` subtracts Int64 offsets and forms the linear index with Int64 multiplies by runtime strides, which
##### the GPU executes as several 32-bit integer instructions each (on its half-rate integer pipe). In stencil-heavy
##### kernels this address arithmetic is a quarter to a third of all instructions. `convert_to_device` therefore
##### replaces the `CuDeviceArray` parent of every `OffsetArray` kernel argument (fields, grid metrics, and any
##### struct Adapt traverses) by an `Int32IndexedArray`: the same pointer and element type with `Int32` dimensions,
##### whose linear index is computed in `Int32` and widened once for the address. The memory and the values are the
##### same; only the integer instructions change. The `OffsetArray` wrapper stays, so offsets, axes, `parent` and
##### linear indexing keep their semantics. An argument array of `typemax(Int32)` or more elements (its linear index
##### needs 64 bits) raises an `ArgumentError` naming the switch: the compile-time preference `AC.INT32_INDEXING`
##### (`int32_indexing = false` in LocalPreferences.toml) restores 64-bit indexing everywhere. The conversion is
##### type-stable either way; the length check reads the host-side array descriptor and allocates nothing.
#####

"""
    Int32IndexedArray{T, N, A}

A device array (the parent of an `OffsetArray` kernel argument) whose index arithmetic is done in `Int32`:
the pointer of a `CuDeviceArray` and its dimensions as `Int32`. Built by `convert_to_device` only for arrays of
fewer than `typemax(Int32)` elements.
"""
struct Int32IndexedArray{T, N, A} <: DenseArray{T, N}
    ptr :: Core.LLVMPtr{T, A}
    dims :: NTuple{N, Int32}
end

Base.size(a::Int32IndexedArray) = map(Int, a.dims)
Base.IndexStyle(::Type{<:Int32IndexedArray}) = IndexLinear()
Base.elsize(::Type{<:Int32IndexedArray{T}}) where T = sizeof(T)
Base.pointer(a::Int32IndexedArray{T}, i::Integer = 1) where T = a.ptr + (Int(i) - 1) * sizeof(T)
Base.unsafe_convert(::Type{Core.LLVMPtr{T, A}}, a::Int32IndexedArray{T, <:Any, A}) where {T, A} = a.ptr

# The device form stands in for a `CuDeviceArray`, so it answers `architecture` as `CuDeviceArray` does (above). Host
# code that receives converted arguments asks for it: `update_pole_value!` launches on `architecture(bc.data)` of the
# device-form PolarValue that `fill_halo_regions!(::Field)` passes after `convert_to_device` (reached by a Float64 grid
# regridding Float32 ETOPO topography on a global source grid).
AC.architecture(::Int32IndexedArray) = CUDAGPU()
AC.architecture(::Type{<:Int32IndexedArray}) = CUDAGPU()

# 0-based linear index of the 1-based indices `I`, in Int32: (i₁ - 1) + d₁ ((i₂ - 1) + d₂ (…)).
@inline int32_linear_index(::Tuple{}, ::Tuple{}) = Int32(0)
@inline int32_linear_index(dims::Tuple, I::Tuple) =
    ((first(I) % Int32) - Int32(1)) + first(dims) * int32_linear_index(Base.tail(dims), Base.tail(I))

# The element at 0-based linear index ℓ: `unsafe_load(ptr, n)` addresses ptr + (n - 1), so n = Int(ℓ) + 1 folds to
# a single sign extension of ℓ. Alignment as `CuDeviceArray` (`Base.datatype_alignment`).
@inline load_int32_indexed(a::Int32IndexedArray{T}, ℓ::Int32) where T =
    unsafe_load(a.ptr, Int(ℓ) + 1, Val(Base.datatype_alignment(T)))
@inline store_int32_indexed!(a::Int32IndexedArray{T}, x, ℓ::Int32) where T =
    unsafe_store!(a.ptr, convert(T, x), Int(ℓ) + 1, Val(Base.datatype_alignment(T)))

# A simplified bounds check as `CuDeviceArray`'s (on the linear index; elided under `@inbounds`).
@inline in_int32_bounds(a::Int32IndexedArray, ℓ::Int32) = (ℓ % UInt32) < (prod(a.dims) % UInt32)

Base.@propagate_inbounds function Base.getindex(a::Int32IndexedArray, i::Integer)
    ℓ = (i % Int32) - Int32(1)
    @boundscheck in_int32_bounds(a, ℓ) || Base.throw_boundserror(a, (i,))
    return load_int32_indexed(a, ℓ)
end

Base.@propagate_inbounds function Base.getindex(a::Int32IndexedArray{T, N}, I::Vararg{Integer, N}) where {T, N}
    ℓ = int32_linear_index(a.dims, I)
    @boundscheck in_int32_bounds(a, ℓ) || Base.throw_boundserror(a, I)
    return load_int32_indexed(a, ℓ)
end

Base.@propagate_inbounds function Base.setindex!(a::Int32IndexedArray, x, i::Integer)
    ℓ = (i % Int32) - Int32(1)
    @boundscheck in_int32_bounds(a, ℓ) || Base.throw_boundserror(a, (i,))
    store_int32_indexed!(a, x, ℓ)
    return a
end

Base.@propagate_inbounds function Base.setindex!(a::Int32IndexedArray{T, N}, x, I::Vararg{Integer, N}) where {T, N}
    ℓ = int32_linear_index(a.dims, I)
    @boundscheck in_int32_bounds(a, ℓ) || Base.throw_boundserror(a, I)
    store_int32_indexed!(a, x, ℓ)
    return a
end

# One index on a vector: the linear methods (disambiguates the two above).
Base.@propagate_inbounds function Base.getindex(a::Int32IndexedArray{T, 1}, i::Integer) where T
    ℓ = (i % Int32) - Int32(1)
    @boundscheck in_int32_bounds(a, ℓ) || Base.throw_boundserror(a, (i,))
    return load_int32_indexed(a, ℓ)
end

Base.@propagate_inbounds function Base.setindex!(a::Int32IndexedArray{T, 1}, x, i::Integer) where T
    ℓ = (i % Int32) - Int32(1)
    @boundscheck in_int32_bounds(a, ℓ) || Base.throw_boundserror(a, (i,))
    store_int32_indexed!(a, x, ℓ)
    return a
end

struct Int32Indexing end

@noinline function throw_int32_indexing_error(len)
    throw(ArgumentError("a kernel argument array has $len elements, more than 32-bit index arithmetic can address " *
                        "($(typemax(Int32))); set the Oceananigans preference `int32_indexing = false` " *
                        "(LocalPreferences.toml) to use 64-bit indices."))
end

"""
    int32_indexed_parent(p)

The parent an `OffsetArray` kernel argument gets on the device: an [`Int32IndexedArray`](@ref) over the memory of
the `CuDeviceArray` `p` (an `ArgumentError` if `p` has `typemax(Int32)` or more elements), any other parent as is.
"""
int32_indexed_parent(p) = p

@inline function int32_indexed_parent(p::CuDeviceArray{T, N, A}) where {T, N, A}
    length(p) < typemax(Int32) || throw_int32_indexing_error(length(p))
    return Int32IndexedArray{T, N, A}(pointer(p), map(Int32, size(p)))
end

function Adapt.adapt_structure(::Int32Indexing, a::OffsetArray{T, N, <:CuDeviceArray}) where {T, N}
    p = int32_indexed_parent(parent(a))
    return OffsetArray{T, N, typeof(p)}(p, a.offsets)
end

# The kernel arguments with 32-bit index arithmetic (`AC.INT32_INDEXING`, a compile-time constant), or as converted.
@inline int32_indexed(x) = AC.INT32_INDEXING ? Adapt.adapt(Int32Indexing(), x) : x

# Whether a type (an argument type, recursively through its fields) may hold a raw pointer, whose
# `cudaconvert` is not the identity (`CuPtr` → `LLVMPtr`).
holds_raw_pointer(::Type{<:Union{Ptr, CuPtr}}) = true
function holds_raw_pointer(T::Type)
    isconcretetype(T) || return true
    return any(holds_raw_pointer, fieldtypes(T))
end

# An argument is already in its device form when its type is `isbits`, holds no raw pointer and is the type
# `cudaconvert` gives it back as (a host object that is `isbits` but converts to another type — a regular
# `RectilinearGrid`, whose architecture becomes `nothing` — is not). Decided on the type, at compile time
# (`Core.Compiler.return_type` of a concretely inferred call folds to a constant). Such an argument passes
# through untouched, so arguments cached in their device form (Breeze's kernel-argument cache) cost nothing
# to convert again; every other argument is converted with `cudaconvert`.
@generated has_no_raw_pointer(::Type{T}) where T = !holds_raw_pointer(T)

@inline is_device_form_type(::Type{T}) where T =
    isbitstype(T) && has_no_raw_pointer(T) && (Core.Compiler.return_type(CUDA.cudaconvert, Tuple{T}) === T)

# Inference cannot always type `cudaconvert` of an `isbits`, pointer-free argument (a closure whose captured values
# Adapt rebuilds through a call inference gives up on: `return_type` is then not concrete). The compile-time test
# above says no for such a type although `cudaconvert` returns its values unchanged, and every launch that takes
# such an argument would miss the cached launch. For exactly those types the question is settled once per type,
# from the first value seen: the type is a device form when `cudaconvert` gives that value back with its type.
# (`CHECK_CACHED_KERNEL_LAUNCH[] = true` asserts `cudaconvert(args) === args` on every cached launch.) A type
# whose conversion infers to another concrete type (a host object) is never asked.
@inline function undecided_device_form_type(::Type{T}) where T
    (isbitstype(T) && has_no_raw_pointer(T)) || return false
    C = Core.Compiler.return_type(CUDA.cudaconvert, Tuple{T})
    return !(C === T) && !isconcretetype(C)
end

const observed_device_forms = IdDict{Any, Bool}()
const observed_device_form_lock = ReentrantLock()

@noinline function observed_device_form(x::T) where T
    observed = @lock observed_device_form_lock get(observed_device_forms, T, nothing)
    if observed === nothing
        observed = typeof(CUDA.cudaconvert(x)) === T
        @lock observed_device_form_lock observed_device_forms[T] = observed
    end
    return observed::Bool
end

# Whether the argument `x` is in its device form: decided on its type at compile time, or, where that is
# undecided, from the per-type observation.
@inline is_device_form(x::T) where T =
    is_device_form_type(T) || (undecided_device_form_type(T) && observed_device_form(x))

# The conversion path gives the converted arrays 32-bit index arithmetic (`int32_indexed`, above); its result is in
# device form (an `OffsetArray` over an `Int32IndexedArray` is `isbits`, pointer-free for this test and returned
# unchanged by `cudaconvert`), so converting it again passes it through.
@inline device_form(x::T) where T = is_device_form(x) ? x : int32_indexed(CUDA.cudaconvert(x))

@inline AC.convert_to_device(::CUDAGPU, args) = device_form(args)
@inline AC.convert_to_device(::CUDAGPU, args::Tuple) = map(device_form, args)

BC.validate_boundary_condition_architecture(::CuArray, ::AC.GPU, bc, side) = nothing

BC.validate_boundary_condition_architecture(::CuArray, ::AC.CPU, bc, side) =
    throw(ArgumentError("$side $bc must use `Array` rather than `CuArray` on CPU architectures!"))

function SO.plan_forward_transform(A::CuArray, ::Union{GD.Bounded, GD.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return cuFFT.plan_fft!(A, dims)
end

FD.set!(v::FD.Field, a::CuArray) = FD.set_to_array!(v, a)
FD.set!(v::DC.DistributedField, a::CuArray) = FD.set_to_array!(v, a)
FD.set!(v::MR.MultiRegionField, a::CuArray) = apply_regionally!(FD.set!, v, a)

function SO.plan_backward_transform(A::CuArray, ::Union{GD.Bounded, GD.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return cuFFT.plan_ifft!(A, dims)
end

# CUDA version, the indices are passed implicitly
# You must not use KA here as this code is executed in another scope
CUDA.@device_override @inline function __validindex(ctx::MappedCompilerMetadata)
    if __dynamic_checkbounds(ctx)
        index = @inbounds linear_expand(__iterspace(ctx), CUDA.blockIdx().x, CUDA.threadIdx().x)
        return index ≤ __linear_ndrange(ctx)
    else
        return true
    end
end

#####
##### Cached launch of configured kernels (host cost of `launch!` and the halo fills)
#####
#
# `KA.Kernel{CUDABackend}` launches (CUDACore's CUDAKernels.jl) re-convert every argument with
# `cudaconvert` (a closure `map` that is not inferred for Oceananigans' nested argument types: the grid,
# boundary conditions with field time series, field tuples), rebuild the argument tuple type and look up
# the compiled kernel, and the compiled kernel's call converts the arguments once more — on every launch.
# Oceananigans has already converted them (`Architectures.convert_to_device`), so for a kernel with a
# static workgroup and ndrange whose arguments are all in their device form (`is_device_form`), this launches the
# same compiled kernel (`cufunction` with the same function, argument types, `maxthreads` and
# `always_inline`), cached per CUDA context, kernel function, argument types and compile options, with
# the same threads and blocks (computed per launch from the kernel's own static sizes, exactly as
# CUDAKernels.jl does) and the same argument values, without re-converting them
# (`convert = Val(false)`). Launches with an argument that is not in its device form (not `isbits`, a
# `Type`, a raw pointer, or a type `cudaconvert` changes), or kernels with a dynamic workgroup or ndrange,
# take the standard launch. `CACHED_KERNEL_LAUNCH[] = false` disables the cache; `CHECK_CACHED_KERNEL_LAUNCH[] = true`
# asserts on every cached launch that converting the arguments again leaves them bit-identical
# (`cudaconvert(args) === args`), the condition under which the cached launch is the standard one.

const CACHED_KERNEL_LAUNCH = Ref(true)
const CHECK_CACHED_KERNEL_LAUNCH = Ref(false)
const cached_launch_kernels = Dict{Any, Any}()
const cached_launch_lock = ReentrantLock()
# launches that took the cached path / the standard one (diagnostics)
const cached_launch_counts = zeros(Int, 2)
# With `RECORD_STANDARD_LAUNCHES[] = true`: the launches that took the standard path, counted by kernel function
# and reason — `:dynamic` (workgroup or ndrange not static) or the positions of the arguments that are not in
# their device form (1 is the kernel's context) (diagnostics).
const RECORD_STANDARD_LAUNCHES = Ref(false)
const standard_launches = Dict{Any, Int}()

@noinline function record_standard_launch(f, reason)
    key = (f, reason)
    @lock cached_launch_lock standard_launches[key] = get(standard_launches, key, 0) + 1
    return nothing
end

@inline function standard_kernel_launch!(loop!, reason, args...)
    cached_launch_counts[2] += 1
    RECORD_STANDARD_LAUNCHES[] && record_standard_launch(loop!.f, reason(args))
    return loop!(args...)
end

@inline dynamic_launch(args) = :dynamic

# Whether every argument of a launch is in its device form (see `is_device_form`; a `Type` argument is
# not `isbits`, so the argument tuple type then equals the `Tuple{map(Core.Typeof, args)...}` that `@cuda`
# compiles for).
@inline all_device_form(::Tuple{}) = true
@inline all_device_form(args::Tuple) = is_device_form(first(args)) & all_device_form(Base.tail(args))

const StaticNDRange = Union{KA.StaticSize, UT.OffsetStaticSize}

@inline UT.launch_configured_kernel!(loop!::KA.Kernel{CUDABackend}, args...) =
    CACHED_KERNEL_LAUNCH[] ? cached_kernel_launch!(loop!, args...) : loop!(args...)

@inline cached_kernel_launch!(loop!, args...) = standard_kernel_launch!(loop!, dynamic_launch, args...)

# The size of a static launch range, one entry per dimension: a constant of the range's type.
@inline static_launch_size(::Type{KA.StaticSize{S}}) where S = S
@generated static_launch_size(::Type{UT.OffsetStaticSize{S}}) where S = map(r -> r isa Integer ? Int(r) : length(r), S)
@inline static_launch_size(::KA.Kernel{CUDABackend, <:Any, N}) where N <: StaticNDRange = static_launch_size(N)

# The kernel object with the workgroup of its per-kernel launch configuration (`UT.kernel_launch_workgroup`):
# the object itself unless an entry names another workgroup, decided at compile time on the kernel function,
# the configured workgroup and the size of the launch's static range. An entry has at least the dimensions of
# the configured workgroup and at most those of the range (a two-dimensional `(16, 16)` may become `(32, 1, 8)`
# on a three-dimensional range), and at most 1024 threads.
@inline function configured_kernel(obj::KA.Kernel{CUDABackend, KA.StaticSize{W}, N, F}) where {W, N <: StaticNDRange, F}
    L = static_launch_size(N)
    W′ = UT.kernel_launch_workgroup(obj.f, Val(W), Val(L))
    W′ === W && return obj
    (W′ isa Tuple{Vararg{Int}} && length(W) <= length(W′) <= length(L) && all(>(0), W′) && prod(W′) <= 1024) ||
        throw(ArgumentError("kernel_launch_workgroup of $(nameof(F)) must return $(length(W)) to $(length(L)) positive integers " *
                            "with a product of at most 1024 for a launch over $(L), not $(W′)"))
    return KA.Kernel{CUDABackend, KA.StaticSize{W′}, N, F}(KA.backend(obj), obj.f)
end

@inline function cached_kernel_launch!(configured::KA.Kernel{CUDABackend, <:KA.StaticSize, <:StaticNDRange}, args...)
    obj = configured_kernel(configured)
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, nothing, nothing)
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    kernel_args = (ctx, args...)
    all_device_form(kernel_args) ||
        return standard_kernel_launch!(configured, _ -> findall(arg -> !is_device_form(arg), collect(Any, kernel_args)), args...)
    TT = typeof(kernel_args)
    cached_launch_counts[1] += 1

    blocks = length(KA.blocks(iterspace))
    threads = length(KA.workitems(iterspace))
    blocks == 0 && return nothing

    CHECK_CACHED_KERNEL_LAUNCH[] && check_cached_launch_arguments(obj, kernel_args)

    maxthreads = prod(KA.get(KA.workgroupsize(obj)))
    always_inline = KA.backend(obj).always_inline
    maxregs = UT.kernel_launch_maxregs(obj.f, Val(static_launch_size(obj)))
    kernel = cached_host_kernel(obj.f, TT, maxthreads, always_inline, maxregs)
    kernel(kernel_args...; threads, blocks, convert = Val(false))
    return nothing
end

# (the workgroup is part of `TT` through the kernel's context type, and of `maxthreads`)
function cached_host_kernel(f::F, ::Type{TT}, maxthreads, always_inline, maxregs) where {F, TT}
    key = (context(), f, TT, maxthreads, always_inline, maxregs)
    kernel = @lock cached_launch_lock get(cached_launch_kernels, key, nothing)
    if kernel === nothing
        kernel = maxregs > 0 ? CUDA.cufunction(f, TT; always_inline, maxthreads, maxregs) :
                               CUDA.cufunction(f, TT; always_inline, maxthreads)
        @lock cached_launch_lock cached_launch_kernels[key] = kernel
    end
    return kernel::CUDA.HostKernel{F, TT}
end

@noinline function check_cached_launch_arguments(obj, kernel_args)
    reconverted = map(CUDA.cudaconvert, kernel_args)
    if !(reconverted === kernel_args)
        changed = [string(i, ": ", typeof(kernel_args[i]), " → ", typeof(reconverted[i]))
                   for i in eachindex(kernel_args) if !(reconverted[i] === kernel_args[i])]
        error("cached kernel launch of $(nameof(typeof(obj.f))): the arguments are not in their device form " *
              "(cudaconvert changes them); the cached launch would differ from the standard one. Changed: " *
              join(changed, "; "))
    end
    return nothing
end

@inline UT.sync_device!(::CuDevice)      = CUDA.synchronize()
@inline UT.sync_device!(::CUDAGPU)       = CUDA.synchronize()
@inline UT.sync_device!(::CUDABackend)   = CUDA.synchronize()

# MPI needs the event's context bound to the thread that posts (for CUDA IPC)
struct ContextEvent
    event :: CUDA.CuEvent
    context :: CUDA.CuContext
end

DC.new_event(::CUDAGPU) = ContextEvent(CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING), CUDA.context())
DC.record_event!(event::ContextEvent, ::CUDAGPU) = CUDA.record(event.event, CUDA.stream())
DC.bind_thread_to_device!(event::ContextEvent) = CUDA.activate(event.context)

# Same as `CUDA.isdone`, without the allocation of its GC-safe ccall: the progress worker polls this in a loop,
# and `cuEventQuery` never blocks. The driver bindings live in `CUDA` on CUDA.jl v5 and in `CUDACore` on v6.
const CUDADriver = parentmodule(CUDA.isdone)

function DC.event_done(event::ContextEvent)
    result = ccall((:cuEventQuery, CUDADriver.libcuda), CUDADriver.CUresult, (CUDADriver.CUevent,), event.event)
    result == CUDADriver.ERROR_NOT_READY && return false
    result == CUDADriver.SUCCESS && return true
    return CUDADriver.throw_api_error(result)
end

#####
##### Field data with static dimensions inside kernels
#####

# Fields reach kernels as device arrays whose dimensions are type parameters. Indexing an array
# whose dimensions are only known at run time needs a separate 64-bit address register for every
# stencil offset in y and z (the strides are run-time values); with static dimensions the offsets
# are compile-time constants that become immediate offsets of the load instructions. High-order
# advection stencils read tens of neighbors, so this lowers register pressure (≈ 75 to ≈ 50
# registers per thread for WENO tendency kernels) and raises occupancy. Kernels are therefore
# specialized on the array sizes, as they are under Reactant.
struct StaticSizeDeviceArray{T, N, D} <: DenseArray{T, N}
    ptr :: Core.LLVMPtr{T, CUDA.AS.Global}
end

# Convert through CUDA's own `CuDeviceArray` conversion, which also prefetches unified memory
function StaticSizeDeviceArray(to::CUDA.KernelAdaptor, a::CUDA.DenseCuArray{T, N}) where {T, N}
    device_array = Adapt.adapt_storage(to, a)
    return StaticSizeDeviceArray{T, N, size(a)}(pointer(device_array))
end

Base.size(::StaticSizeDeviceArray{T, N, D}) where {T, N, D} = D
Base.length(::StaticSizeDeviceArray{T, N, D}) where {T, N, D} = prod(D)
Base.IndexStyle(::Type{<:StaticSizeDeviceArray}) = IndexLinear()
Base.pointer(a::StaticSizeDeviceArray) = a.ptr
Base.elsize(::Type{<:StaticSizeDeviceArray{T}}) where T = sizeof(T)

@inline Base.getindex(a::StaticSizeDeviceArray{T}, i::Integer) where T =
    unsafe_load(a.ptr, i, Val(Base.datatype_alignment(T)))

@inline function Base.setindex!(a::StaticSizeDeviceArray{T}, x, i::Integer) where T
    unsafe_store!(a.ptr, convert(T, x)::T, i, Val(Base.datatype_alignment(T)))
    return a
end

AC.architecture(::StaticSizeDeviceArray) = CUDAGPU()

const CuOffsetArray{T, N} = OffsetArray{T, N, <:CUDA.DenseCuArray{T, N}}

kernel_field_data(to, data) = Adapt.adapt(to, data)

# Arrays of isbits unions (which `CuDeviceArray` stores with a selector byte array) keep the default
function kernel_field_data(to, data::CuOffsetArray{T}) where T
    isbitstype(T) || return Adapt.adapt(to, data)
    return OffsetArray(StaticSizeDeviceArray(to, parent(data)), data.offsets)
end

Adapt.adapt_structure(to::CUDA.KernelAdaptor, f::FD.Field) = kernel_field_data(to, f.data)

# Reduced fields keep their location (see `Adapt.adapt_structure(to, ::ReducedField)`)
function Adapt.adapt_structure(to::CUDA.KernelAdaptor, reduced_field::FD.ReducedField)
    LX, LY, LZ = FD.location(reduced_field)
    data = kernel_field_data(to, reduced_field.data)
    return FD.Field{LX, LY, LZ}(nothing, data, nothing, nothing, nothing, nothing, nothing)
end

# Use faster versions of `newton_div` on Nvidia GPUs
CUDA.@device_override UT.newton_div(::Type{UT.BackendOptimizedDivision}, a, b) = a * fast_inv_cuda(b)

function fast_inv_cuda(a::Float64)
    # Get the approximate reciprocal
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rcp-approx-ftz-f64
    # This instruction chops off last 32bits of mantissa and computes inverse
    # while treating all subnormal numbers as 0.0
    # If reciprocal would be subnormal, underflows to 0.0
    # 32 least significant bits of the result are filled with 0s
    inv_a = ccall("llvm.nvvm.rcp.approx.ftz.d", llvmcall, Float64, (Float64,), a)

    # Approximate the missing 32bits of mantissa with a single cubic iteration
    e = fma(inv_a, -a, 1.0)
    e = fma(e, e, e)
    inv_a = fma(e, inv_a, inv_a)
    return inv_a
end

function fast_inv_cuda(a::Float32)
    # This instruction just computes reciprocal flushing subnormals to 0.0
    # Hence for subnormal inputs it returns Inf
    # For large number whose reciprocal is subnormal it underflows to 0.0
    inv_a = ccall("llvm.nvvm.rcp.approx.ftz.f", llvmcall, Float32, (Float32,), a)
    return inv_a
end

# Other number types (e.g. BFloat16) have no approximate reciprocal instruction
fast_inv_cuda(a) = inv(a)

end # module OceananigansCUDAExt
