module OceananigansCUDAExt

using InteractiveUtils: versioninfo
using CUDA: CUDA, CuArray, CuContext, CuDevice, CuDeviceArray, CuPtr, context,
    context!, cu, CUDA.CUDAKernels.CUDABackend
using CUDA.CUSPARSE: CuSparseMatrixCSC
using GPUArraysCore: allowscalar
using GPUArrays: unsafe_free!
using Oceananigans.Utils: linear_expand, __linear_ndrange, MappedCompilerMetadata

import Oceananigans.Architectures as AC
import Oceananigans.BoundaryConditions as BC
import Oceananigans.DistributedComputations as DC
import Oceananigans.Fields as FD
import Oceananigans.Grids as GD
import Oceananigans.Solvers as SO
import Oceananigans.Utils as UT
import SparseArrays: SparseMatrixCSC
import KernelAbstractions: __iterspace, __dynamic_checkbounds, __validindex
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
AC.on_architecture(::CUDAGPU, a::SubArray{<:Any, <:Any, <:CuArray}) = a
AC.on_architecture(::CUDAGPU, a::SubArray{<:Any, <:Any, <:Array}) = CuArray(a)
AC.on_architecture(::AC.CPU, a::SubArray{<:Any, <:Any, <:CuArray}) = Array(a)
AC.on_architecture(::CUDAGPU, a::StepRangeLen) = a
AC.on_architecture(arch::Distributed, a::CuArray) = AC.on_architecture(AC.child_architecture(arch), a)
AC.on_architecture(arch::Distributed, a::SubArray{<:Any, <:Any, <:CuArray}) = AC.on_architecture(child_architecture(arch), a)

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

@inline AC.unsafe_free!(a::CuArray) = unsafe_free!(a)

@inline AC.convert_to_device(::CUDAGPU, args) = CUDA.cudaconvert(args)
@inline AC.convert_to_device(::CUDAGPU, args::Tuple) = map(CUDA.cudaconvert, args)

BC.validate_boundary_condition_architecture(::CuArray, ::AC.GPU, bc, side) = nothing

BC.validate_boundary_condition_architecture(::CuArray, ::AC.CPU, bc, side) =
    throw(ArgumentError("$side $bc must use `Array` rather than `CuArray` on CPU architectures!"))

function SO.plan_forward_transform(A::CuArray, ::Union{GD.Bounded, GD.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return CUDA.CUFFT.plan_fft!(A, dims)
end

FD.set!(v::FD.Field, a::CuArray) = FD.set_to_array!(v, a)
FD.set!(v::DC.DistributedField, a::CuArray) = FD.set_to_array!(v, a)

function SO.plan_backward_transform(A::CuArray, ::Union{GD.Bounded, GD.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return CUDA.CUFFT.plan_ifft!(A, dims)
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

@inline UT.sync_device!(::CuDevice)      = CUDA.synchronize()
@inline UT.sync_device!(::CUDAGPU)       = CUDA.synchronize()
@inline UT.sync_device!(::CUDABackend)   = CUDA.synchronize()

# Use faster versions of `newton_div` on Nvidia GPUs
CUDA.@device_override UT.newton_div(::Type{Float32}, a, b::Float64) = a * fast_inv_cuda(b)
CUDA.@device_override UT.newton_div(::Type{Float64}, a, b::Float64) = a * fast_inv_cuda(b)

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

end # module OceananigansCUDAExt
