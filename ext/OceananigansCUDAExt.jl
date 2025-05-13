module OceananigansCUDAExt

using Oceananigans
using CUDA, CUDA.CUSPARSE, CUDA.CUFFT
using KernelAbstractions
import Oceananigans.Architectures as AC
import Oceananigans.BoundaryConditions as BC
import Oceananigans.Solvers as SO
import Oceananigans.Utils as UT
import SparseArrays: SparseMatrixCSC
import KernelAbstractions: __iterspace, __groupindex, __dynamic_checkbounds,
                           __validindex, CompilerMetadata
import Oceananigans.DistributedComputations: Distributed

const GPUVar = Union{CuArray, CuContext, CuPtr, Ptr}

function __init__()
    if CUDA.has_cuda()
        @debug "CUDA-enabled GPU(s) detected:"
        for (gpu, dev) in enumerate(CUDA.devices())
            @debug "$dev: $(CUDA.name(dev))"
        end

        CUDA.allowscalar(false)
    end
end

const CUDAGPU = AC.GPU{<:CUDABackend}

CUDAGPU() = GPU(CUDABackend(always_inline=true))

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


Base.summary(::CUDAGPU) = "CUDAGPU"

AC.architecture(::CuArray) = CUDAGPU()
AC.architecture(::CuSparseMatrixCSC) = AC.GPU()
AC.architecture(::SparseMatrixCSC)   = AC.CPU()
AC.array_type(::AC.GPU{CUDABackend}) = CuArray

AC.on_architecture(::AC.CPU, a::CuArray) = Array(a)

AC.on_architecture(::CUDAGPU, a::Array) = CuArray(a)
AC.on_architecture(::CUDAGPU, a::CuArray) = a
AC.on_architecture(::CUDAGPU, a::BitArray) = CuArray(a)
AC.on_architecture(::CUDAGPU, a::SubArray{<:Any, <:Any, <:CuArray}) = a
AC.on_architecture(::CUDAGPU, a::SubArray{<:Any, <:Any, <:Array}) = CuArray(a)
AC.on_architecture(::AC.CPU, a::SubArray{<:Any, <:Any, <:CuArray}) = Array(a)
AC.on_architecture(::CUDAGPU, a::StepRangeLen) = a

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

@inline AC.unsafe_free!(a::CuArray) = CUDA.unsafe_free!(a)

@inline AC.constructors(::AC.GPU{CUDABackend}, A::SparseMatrixCSC) = (CuArray(A.colptr), CuArray(A.rowval), CuArray(A.nzval),  (A.m, A.n))
@inline AC.constructors(::AC.CPU, A::CuSparseMatrixCSC) = (A.dims[1], A.dims[2], Int64.(Array(A.colPtr)), Int64.(Array(A.rowVal)), Array(A.nzVal))
@inline AC.constructors(::AC.GPU{CUDABackend}, A::CuSparseMatrixCSC) = (A.colPtr, A.rowVal, A.nzVal,  A.dims)

@inline AC.unpack_constructors(::AC.CPU, constr::Tuple) = (constr[3], constr[4], constr[5])
@inline AC.copy_unpack_constructors(::AC.CPU, constr::Tuple) = deepcopy((constr[3], constr[4], constr[5]))

@inline AC.arch_sparse_matrix(::AC.GPU{CUDABackend}, constr::Tuple) = CuSparseMatrixCSC(constr...)
@inline AC.arch_sparse_matrix(::AC.CPU, A::CuSparseMatrixCSC)   = SparseMatrixCSC(constructors(AC.CPU(), A)...)
@inline AC.arch_sparse_matrix(::AC.GPU{CUDABackend}, A::SparseMatrixCSC)     = CuSparseMatrixCSC(constructors(AC.GPU(), A)...)

@inline AC.arch_sparse_matrix(::AC.GPU{CUDABackend}, A::CuSparseMatrixCSC) = A

@inline AC.convert_to_device(::CUDAGPU, args) = CUDA.cudaconvert(args)
@inline AC.convert_to_device(::CUDAGPU, args::Tuple) = map(CUDA.cudaconvert, args)


BC.validate_boundary_condition_architecture(::CuArray, ::AC.GPU, bc, side) = nothing

BC.validate_boundary_condition_architecture(::CuArray, ::AC.CPU, bc, side) =
    throw(ArgumentError("$side $bc must use `Array` rather than `CuArray` on CPU architectures!"))

function SO.plan_forward_transform(A::CuArray, ::Union{BC.Bounded, BC.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return CUDA.CUFFT.plan_fft!(A, dims)
end

function SO.plan_backward_transform(A::CuArray, ::Union{BC.Bounded, BC.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return CUDA.CUFFT.plan_ifft!(A, dims)
end

# CUDA version, the indices are passed implicitly
CUDA.@device_override @inline function __validindex(ctx::UT.MappedCompilerMetadata)
    if __dynamic_checkbounds(ctx)
        index = @inbounds UT.linear_expand(__iterspace(ctx), blockIdx().x, threadIdx().x)
        return index ≤ UT.__linear_ndrange(ctx)
    else
        return true
    end
end

@inline UT.sync_device!(::CuDevice)  = UT.sync_device!(CUDABackend())
@inline UT.getdevice(cu::GPUVar, i)     = device(cu)
@inline UT.getdevice(cu::GPUVar)        = device(cu)
@inline UT.switch_device!(dev::CuDevice)            = device!(dev)

AC.on_architecture(arch::Distributed, a::CuArray) = AC.on_architecture(child_architecture(arch), a)
AC.on_architecture(arch::Distributed, a::SubArray{<:Any, <:Any, <:CuArray}) = AC.on_architecture(child_architecture(arch), a)

end # module OceananigansCUDAExt
