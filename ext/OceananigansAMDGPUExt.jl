module OceananigansAMDGPUExt

using AMDGPU
using AMDGPU.rocFFT
using Oceananigans
using Oceananigans.Utils: linear_expand, __linear_ndrange, MappedCompilerMetadata
using KernelAbstractions: __dynamic_checkbounds, __iterspace

import KernelAbstractions: __validindex

import Oceananigans.Architectures:
    architecture,
    convert_to_device,
    on_architecture
import Oceananigans.Solvers:
    plan_backward_transform,
    plan_forward_transform

const ROCGPU = GPU{<:AMDGPU.ROCBackend}
ROCGPU() = GPU(AMDGPU.ROCBackend())

architecture(::ROCArray) = ROCGPU()
Base.summary(::ROCGPU) = "ROCGPU"

on_architecture(::ROCGPU, a::Number) = a
on_architecture(::ROCGPU, a::Array) = ROCArray(a)
on_architecture(::ROCGPU, a::BitArray) = ROCArray(a)
on_architecture(::ROCGPU, a::SubArray{<:Any, <:Any, <:Array}) = ROCArray(a)
on_architecture(::CPU, a::ROCArray) = Array(a)
on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:ROCArray}) = Array(a)
on_architecture(::ROCGPU, a::ROCArray) = a
on_architecture(::ROCGPU, a::SubArray{<:Any, <:Any, <:ROCArray}) = a
on_architecture(::ROCGPU, a::StepRangeLen) = a

@inline convert_to_device(::ROCGPU, args) = AMDGPU.rocconvert(args)
@inline convert_to_device(::ROCGPU, args::Tuple) = map(AMDGPU.rocconvert, args)

function plan_forward_transform(A::ROCArray, ::Union{Bounded, Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return AMDGPU.rocFFT.plan_fft!(A, dims)
end

function plan_backward_transform(A::ROCArray, ::Union{Bounded, Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return AMDGPU.rocFFT.plan_bfft!(A, dims)
end

plan_backward_transform(A::ROCArray, ::Flat, args...) = nothing
plan_forward_transform(A::ROCArray, ::Flat, args...) = nothing

AMDGPU.Device.@device_override @inline function __validindex(ctx::MappedCompilerMetadata)
    if __dynamic_checkbounds(ctx)
        I = @inbounds linear_expand(__iterspace(ctx), AMDGPU.Device.blockIdx().x, AMDGPU.Device.threadIdx().x)
        return I in __linear_ndrange(ctx)
    else
        return true
    end
end

end # module
