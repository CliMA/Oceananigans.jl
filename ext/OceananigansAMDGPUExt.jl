module OceananigansAMDGPUExt

using Oceananigans
using InteractiveUtils
using AMDGPU, AMDGPU.rocSPARSE, AMDGPU.rocFFT
using Oceananigans.Utils: linear_expand, __linear_ndrange, MappedCompilerMetadata
using KernelAbstractions: __dynamic_checkbounds, __iterspace
using KernelAbstractions
import Oceananigans.Architectures as AC
import Oceananigans.BoundaryConditions as BC
import Oceananigans.DistributedComputations as DC
import Oceananigans.Fields as FD
import Oceananigans.Grids as GD
import Oceananigans.Solvers as SO
import Oceananigans.Utils as UT
import SparseArrays: SparseMatrixCSC
import KernelAbstractions: __iterspace, __groupindex, __dynamic_checkbounds,
                           __validindex, CompilerMetadata
import Oceananigans.DistributedComputations: Distributed

const GPUVar = Union{ROCArray, Ptr}

function __init__()
    if AMDGPU.functional()
        @debug "ROCm-enabled GPU(s) detected:"
        for (gpu, dev) in enumerate(AMDGPU.devices())
            @debug "$dev: $(AMDGPU.name(dev))"
        end
    end
end

const ROCGPU = AC.GPU{ROCBackend}
ROCGPU() = AC.GPU(AMDGPU.ROCBackend())

architecture(::ROCArray) = ROCGPU()
Base.summary(::ROCGPU) = "ROCGPU"

AC.architecture(::ROCArray) = ROCGPU()
AC.architecture(::ROCSparseMatrixCSC) = ROCGPU()
AC.array_type(::AC.GPU{ROCBackend}) = ROCArray

AC.on_architecture(::ROCGPU, a::Number) = a
AC.on_architecture(::AC.CPU, a::ROCArray) = Array(a)
AC.on_architecture(::ROCGPU, a::Array) = ROCArray(a)
AC.on_architecture(::ROCGPU, a::ROCArray) = a
AC.on_architecture(::ROCGPU, a::BitArray) = ROCArray(a)
AC.on_architecture(::ROCGPU, a::SubArray{<:Any, <:Any, <:ROCArray}) = a
AC.on_architecture(::ROCGPU, a::SubArray{<:Any, <:Any, <:Array}) = ROCArray(a)
AC.on_architecture(::AC.CPU, a::SubArray{<:Any, <:Any, <:ROCArray}) = Array(a)
AC.on_architecture(::ROCGPU, a::StepRangeLen) = a
AC.on_architecture(arch::Distributed, a::ROCArray) = AC.on_architecture(AC.child_architecture(arch), a)
AC.on_architecture(arch::Distributed, a::SubArray{<:Any, <:Any, <:ROCArray}) = AC.on_architecture(child_architecture(arch), a)

function unified_array(::ROCGPU, a::AbstractArray)
    error("unified_array is not implemented for ROCGPU.")
end

## GPU to GPU copy of contiguous data
@inline function AC.device_copy_to!(dst::ROCArray, src::ROCArray; async::Bool = false)
    if async == true
        @warn "Asynchronous copy is not supported for ROCArray. Falling back to synchronous copy."
    end
    copyto!(dst, src)
    return dst
end

@inline AC.unsafe_free!(a::ROCArray) = AMDGPU.unsafe_free!(a)

@inline AC.constructors(::AC.GPU{ROCBackend}, A::SparseMatrixCSC) = (ROCArray(A.colptr), ROCArray(A.rowval), ROCArray(A.nzval),  (A.m, A.n))
@inline AC.constructors(::AC.CPU, A::ROCSparseMatrixCSC) = (A.dims[1], A.dims[2], Int64.(Array(A.colPtr)), Int64.(Array(A.rowVal)), Array(A.nzVal))
@inline AC.constructors(::AC.GPU{ROCBackend}, A::ROCSparseMatrixCSC) = (A.colPtr, A.rowVal, A.nzVal,  A.dims)

@inline AC.arch_sparse_matrix(::AC.GPU{ROCBackend}, constr::Tuple) = ROCSparseMatrixCSC(constr...)
@inline AC.arch_sparse_matrix(::AC.CPU, A::ROCSparseMatrixCSC)   = SparseMatrixCSC(AC.constructors(AC.CPU(), A)...)
@inline AC.arch_sparse_matrix(::AC.GPU{ROCBackend}, A::SparseMatrixCSC)     = ROCSparseMatrixCSC(AC.constructors(AC.GPU(), A)...)
@inline AC.arch_sparse_matrix(::AC.GPU{ROCBackend}, A::ROCSparseMatrixCSC) = A

@inline convert_to_device(::ROCGPU, args) = AMDGPU.rocconvert(args)
@inline convert_to_device(::ROCGPU, args::Tuple) = map(AMDGPU.rocconvert, args)


BC.validate_boundary_condition_architecture(::ROCArray, ::AC.GPU, bc, side) = nothing

BC.validate_boundary_condition_architecture(::ROCArray, ::AC.CPU, bc, side) =
    throw(ArgumentError("$side $bc must use `Array` rather than `ROCArray` on CPU architectures!"))

function SO.plan_forward_transform(A::ROCArray, ::Union{GD.Bounded, GD.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return AMDGPU.rocFFT.plan_fft!(A, dims)
end

FD.set!(v::Field, a::ROCArray) = FD._set!(v, a)
DC.set!(v::DC.DistributedField, a::ROCArray) = DC._set!(v, a)

function SO.plan_backward_transform(A::ROCArray, ::Union{GD.Bounded, GD.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return AMDGPU.rocFFT.plan_ifft!(A, dims)
end

AMDGPU.Device.@device_override @inline function __validindex(ctx::MappedCompilerMetadata)
    if __dynamic_checkbounds(ctx)
        I = @inbounds linear_expand(__iterspace(ctx), AMDGPU.Device.blockIdx().x, AMDGPU.Device.threadIdx().x)
        return I in __linear_ndrange(ctx)
    else
        return true
    end
end

@inline UT.sync_device!(dev::Int64)  = AMDGPU.synchronize()
@inline UT.getdevice(roc::GPUVar, i)     = device(roc)
@inline UT.getdevice(roc::GPUVar)        = device(roc)
@inline UT.switch_device!(dev::Int64)            = device!(dev)
@inline UT.sync_device!(::ROCGPU)      = AMDGPU.synchronize()
@inline UT.sync_device!(::ROCBackend)      = AMDGPU.synchronize()

end # module
