module OceananigansOneAPIExt

using InteractiveUtils: versioninfo
using oneAPI
using oneAPI: method_table # required by `oneAPI.@device_override`
using oneAPI.oneMKL: oneSparseMatrixCSC
using AbstractFFTs: plan_fft!, plan_ifft!

using Oceananigans.Utils: linear_expand, __linear_ndrange, MappedCompilerMetadata

import Oceananigans.Architectures as AC
import Oceananigans.BoundaryConditions as BC
import Oceananigans.DistributedComputations as DC
import Oceananigans.Fields as FD
import Oceananigans.Grids as GD
import Oceananigans.Solvers as SO
import Oceananigans.Utils as UT
import Oceananigans.DistributedComputations: Distributed
import KernelAbstractions: __iterspace, __dynamic_checkbounds, __validindex
import SparseArrays: SparseMatrixCSC

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

function UT.versioninfo_with_gpu(::ONEGPU)
    s = sprint(versioninfo)
    gpu_name = oneAPI.properties(oneAPI.device()).name
    return "oneAPI GPU: $gpu_name"
end

Base.summary(::ONEGPU) = "ONEGPU"
AC.device!(::ONEGPU, i) = oneAPI.device!(i + 1) # oneAPI devices are numbered 1..ndevices

AC.architecture(::oneArray) = ONEGPU()
AC.architecture(::Type{oneArray}) = ONEGPU()
AC.architecture(::oneSparseMatrixCSC) = ONEGPU()
AC.array_type(::AC.GPU{oneAPI.oneAPIBackend}) = oneArray

AC.on_architecture(::ONEGPU, a::Number) = a
AC.on_architecture(::AC.CPU, a::oneArray) = Array(a)
AC.on_architecture(::ONEGPU, a::Array) = oneArray(a)
AC.on_architecture(::ONEGPU, a::oneArray) = a
AC.on_architecture(::ONEGPU, a::BitArray) = oneArray(a)
AC.on_architecture(::ONEGPU, a::StepRangeLen) = a
AC.on_architecture(arch::Distributed, a::oneArray) = AC.on_architecture(AC.child_architecture(arch), a)

@inline AC.sparse_matrix_constructors(::AC.GPU{oneAPI.oneAPIBackend}, A::SparseMatrixCSC) = (oneArray(A.colptr), oneArray(A.rowval), oneArray(A.nzval),  (A.m, A.n))
@inline AC.sparse_matrix_constructors(::AC.CPU, A::oneSparseMatrixCSC) = (A.dims[1], A.dims[2], Int64.(Array(A.colPtr)), Int64.(Array(A.rowVal)), Array(A.nzVal))
@inline AC.sparse_matrix_constructors(::AC.GPU{oneAPI.oneAPIBackend}, A::oneSparseMatrixCSC) = (A.colPtr, A.rowVal, A.nzVal,  A.dims)

@inline AC.sparse_matrix(::AC.GPU{oneAPI.oneAPIBackend}, constr::Tuple) = oneSparseMatrixCSC(constr...)

@inline AC.on_architecture(::AC.CPU, A::oneSparseMatrixCSC)                       = SparseMatrixCSC(AC.sparse_matrix_constructors(AC.CPU(), A)...)
@inline AC.on_architecture(::AC.GPU{oneAPI.oneAPIBackend}, A::SparseMatrixCSC)    = oneSparseMatrixCSC(AC.sparse_matrix_constructors(AC.GPU(), A)...)
@inline AC.on_architecture(::AC.GPU{oneAPI.oneAPIBackend}, A::oneSparseMatrixCSC) = A

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

@inline AC.unsafe_free!(a::oneArray) = oneAPI.unsafe_free!(a)

@inline AC.convert_to_device(::ONEGPU, args) = oneAPI.kernel_convert(args)
@inline AC.convert_to_device(::ONEGPU, args::Tuple) = map(oneAPI.kernel_convert, args)

BC.validate_boundary_condition_architecture(::oneArray, ::AC.GPU, bc, side) = nothing

BC.validate_boundary_condition_architecture(::oneArray, ::AC.CPU, bc, side) =
    throw(ArgumentError("$side $bc must use `Array` rather than `oneArray` on CPU architectures!"))

function SO.plan_forward_transform(A::oneArray, ::Union{GD.Bounded, GD.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return plan_fft!(A, dims)
end

FD.set!(v::FD.Field, a::oneArray) = FD.set_to_array!(v, a)
DC.set!(v::DC.DistributedField, a::oneArray) = DC.set_to_array!(v, a)

function SO.plan_backward_transform(A::oneArray, ::Union{GD.Bounded, GD.Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return plan_ifft!(A, dims)
end

# oneAPI version, the indices are passed implicitly
# You must not use KA here as this code is executed in another scope
oneAPI.@device_override @inline function __validindex(ctx::MappedCompilerMetadata)
    if __dynamic_checkbounds(ctx)
        index = @inbounds linear_expand(__iterspace(ctx), oneAPI.get_group_id(), oneAPI.get_local_id())
        return index ≤ __linear_ndrange(ctx)
    else
        return true
    end
end

@inline UT.sync_device!(::ONEGPU)               = oneAPI.synchronize()
@inline UT.sync_device!(::oneAPI.oneAPIBackend) = oneAPI.synchronize()

end # module
