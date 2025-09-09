module OceananigansOneAPIExt

using Oceananigans
using InteractiveUtils
using oneAPI
using oneAPI.oneMKL: oneSparseMatrixCSR
using oneAPI.oneMKL.FFT
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

const GPUVar = Union{oneArray, Ptr}

function __init__()
    if oneAPI.functional()
        @debug "oneAPI-enabled GPU(s) detected:"
        for (gpu, dev) in enumerate(oneAPI.devices())
            @debug "$dev: $(oneAPI.properties(dev).name)"
        end
    end
end

const ONEGPU = AC.GPU{oneAPI.oneAPIBackend}
ONEGPU() = AC.GPU(oneAPI.oneAPIBackend())

# Default oneAPI backend
function AC.GPU()
    if oneAPI.functional()
        return ONEGPU()
    else
        msg = """We cannot make a GPU with the oneAPI backend:
                 a oneAPI GPU was not found!"""
        throw(ArgumentError(msg))
    end
end

Base.summary(::ONEGPU) = "ONEGPU"

AC.architecture(::oneArray) = ONEGPU()
AC.architecture(::Type{oneArray}) = ONEGPU()
AC.architecture(::oneSparseMatrixCSR) = ONEGPU()
AC.array_type(::AC.GPU{oneAPI.oneAPIBackend}) = oneArray

AC.on_architecture(::ONEGPU, a::Number) = a
AC.on_architecture(::AC.CPU, a::oneArray) = Array(a)
AC.on_architecture(::ONEGPU, a::Array) = oneArray(a)
AC.on_architecture(::ONEGPU, a::oneArray) = a
AC.on_architecture(::ONEGPU, a::BitArray) = oneArray(a)
AC.on_architecture(::ONEGPU, a::SubArray{<:Any, <:Any, <:oneArray}) = a
AC.on_architecture(::ONEGPU, a::SubArray{<:Any, <:Any, <:Array}) = oneArray(a)
AC.on_architecture(::AC.CPU, a::SubArray{<:Any, <:Any, <:oneArray}) = Array(a)
AC.on_architecture(::ONEGPU, a::StepRangeLen) = a
AC.on_architecture(arch::Distributed, a::oneArray) = AC.on_architecture(AC.child_architecture(arch), a)
AC.on_architecture(arch::Distributed, a::SubArray{<:Any, <:Any, <:oneArray}) = AC.on_architecture(AC.child_architecture(arch), a)

function AC.unified_array(::ONEGPU, a::AbstractArray)
    error("unified_array is not implemented for ONEGPU.")
end

## GPU to GPU copy of contiguous data
@inline function AC.device_copy_to!(dst::oneArray, src::oneArray; async::Bool = false)
    if async == true
        @warn "Asynchronous copy is not supported for oneArray. Falling back to synchronous copy."
    end
    copyto!(dst, src)
    return dst
end

@inline function AC.unsafe_free!(a::oneArray)
    # oneArray uses automatic memory management with finalizers
    # So we don't need to explicitly free memory
    nothing
end

@inline AC.constructors(::AC.GPU{oneAPI.oneAPIBackend}, A::SparseMatrixCSC) = A  # oneAPI.jl expects SparseMatrixCSC directly
@inline AC.constructors(::AC.CPU, A::oneSparseMatrixCSR) = SparseMatrixCSC(A)  # Use oneAPI.jl's built-in conversion
@inline AC.constructors(::AC.GPU{oneAPI.oneAPIBackend}, A::oneSparseMatrixCSR) = SparseMatrixCSC(A)  # Use oneAPI.jl's built-in conversion

@inline AC.arch_sparse_matrix(::AC.GPU{oneAPI.oneAPIBackend}, A::SparseMatrixCSC) = oneSparseMatrixCSR(A)
@inline AC.arch_sparse_matrix(::AC.CPU, A::oneSparseMatrixCSR) = SparseMatrixCSC(A)
@inline AC.arch_sparse_matrix(::AC.GPU{oneAPI.oneAPIBackend}, A::oneSparseMatrixCSR) = A

@inline AC.convert_to_device(::ONEGPU, args) = oneAPI.kernel_convert(args)
@inline AC.convert_to_device(::ONEGPU, args::Tuple) = map(oneAPI.kernel_convert, args)

BC.validate_boundary_condition_architecture(::oneArray, ::AC.GPU, bc, side) = nothing

BC.validate_boundary_condition_architecture(::oneArray, ::AC.CPU, bc, side) =
    throw(ArgumentError("$side $bc must use `Array` rather than `oneArray` on CPU architectures!"))

function SO.plan_forward_transform(A::oneArray, ::Union{GD.Bounded, GD.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return return oneAPI.oneMKL.FFT.plan_fft!(A, dims)
end

FD.set!(v::Field, a::oneArray) = FD.set_to_array!(v, a)
DC.set!(v::DC.DistributedField, a::oneArray) = DC.set_to_array!(v, a)

function SO.plan_backward_transform(A::oneArray, ::Union{GD.Bounded, GD.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return oneAPI.oneMKL.FFT.plan_ifft!(A, dims)
end

# oneAPI version of __validindex, similar to CUDA and AMDGPU
oneAPI.@device_override @inline function __validindex(ctx::MappedCompilerMetadata)
    if __dynamic_checkbounds(ctx)
        I = @inbounds linear_expand(__iterspace(ctx), oneAPI.get_group_id(), oneAPI.get_local_id())
        return I in __linear_ndrange(ctx)
    else
        return true
    end
end

@inline UT.getdevice(one::GPUVar, i) = oneAPI.device(one)
@inline UT.getdevice(one::GPUVar) = oneAPI.device(one)
@inline UT.switch_device!(dev::oneAPI.ZeDevice) = oneAPI.device!(dev)
@inline UT.sync_device!(::ONEGPU) = oneAPI.synchronize()
@inline UT.sync_device!(::oneAPI.oneAPIBackend) = oneAPI.synchronize()

function UT.versioninfo_with_gpu(::ONEGPU)
    s = sprint(versioninfo)
    gpu_name = oneAPI.properties(oneAPI.device()).name
    return "oneAPI GPU: $gpu_name"
end

end # module
