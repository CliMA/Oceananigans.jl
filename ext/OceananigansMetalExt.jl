module OceananigansMetalExt

using Metal
using Metal: thread_position_in_threadgroup_1d, threadgroup_position_in_grid_1d
using Oceananigans
using Oceananigans.Utils: linear_expand, __linear_ndrange, MappedCompilerMetadata
using KernelAbstractions: __dynamic_checkbounds, __iterspace
using AbstractFFTs: plan_fft!, plan_ifft!

import KernelAbstractions: __validindex
import Oceananigans.Architectures:
    architecture,
    convert_to_device,
    on_architecture

import Oceananigans.Fields as FD
import Oceananigans.Grids as GD

using Oceananigans.Architectures: Architectures
using Oceananigans.Grids: Bounded, Periodic
using Oceananigans.Solvers: Solvers

const MetalGPU = GPU{<:Metal.MetalBackend}
MetalGPU() = GPU(Metal.MetalBackend())
Base.summary(::MetalGPU) = "MetalGPU"

architecture(::MtlArray) = MetalGPU()
architecture(::Type{MtlArray}) = MetalGPU()

Architectures.array_type(::MetalGPU) = MtlArray

on_architecture(::MetalGPU, a::Number) = a
on_architecture(::MetalGPU, a::Array) = MtlArray(a)
on_architecture(::MetalGPU, a::BitArray) = MtlArray(a)
on_architecture(::CPU, a::MtlArray) = Array(a)
on_architecture(::MetalGPU, a::MtlArray) = a

# Convert StepRangeLen with ref/step::Float64 to ref/step::Float32 for Metal architecture
function on_architecture(::MetalGPU, s::StepRangeLen{FT, Float64, Float64}) where FT
    ref = convert(Float32, s.ref)
    step = convert(Float32, s.step)
    len = s.len
    offset = s.offset
    return StepRangeLen{FT}(ref, step, len, offset)
end

@inline convert_to_device(::MetalGPU, args) = Metal.mtlconvert(args)
@inline convert_to_device(::MetalGPU, args::Tuple) = map(Metal.mtlconvert, args)

Metal.@device_override @inline function __validindex(ctx::MappedCompilerMetadata)
    if __dynamic_checkbounds(ctx)
        index = @inbounds linear_expand(__iterspace(ctx), threadgroup_position_in_grid().x, thread_position_in_threadgroup().x)
        return index ≤ __linear_ndrange(ctx)
    else
        return true
    end
end

function Solvers.plan_forward_transform(A::MtlArray, ::Union{Bounded, Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return plan_fft!(A, dims)
end

function Solvers.plan_backward_transform(A::MtlArray, ::Union{Bounded, Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return plan_ifft!(A, dims)
end

end # module
