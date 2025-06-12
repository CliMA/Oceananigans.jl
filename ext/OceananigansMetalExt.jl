module OceananigansMetalExt

using Metal
using Oceananigans

using Metal: thread_position_in_threadgroup_1d, threadgroup_position_in_grid_1d
using Oceananigans.Utils: linear_expand, __linear_ndrange, MappedCompilerMetadata
using KernelAbstractions: __dynamic_checkbounds, __iterspace
import KernelAbstractions: __validindex

import Oceananigans.Architectures:
    architecture,
    convert_to_device,
    on_architecture

const MetalGPU = GPU{<:Metal.MetalBackend}
MetalGPU() = GPU(Metal.MetalBackend())
Base.summary(::MetalGPU) = "MetalGPU"

architecture(::MtlArray) = MetalGPU()

on_architecture(::MetalGPU, a::Number) = a
on_architecture(::MetalGPU, a::Array) = MtlArray(a)
on_architecture(::MetalGPU, a::BitArray) = MtlArray(a)
on_architecture(::MetalGPU, a::SubArray{<:Any, <:Any, <:Array}) = MtlArray(a)
on_architecture(::CPU, a::MtlArray) = Array(a)
on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:MtlArray}) = Array(a)
on_architecture(::MetalGPU, a::MtlArray) = a
on_architecture(::MetalGPU, a::SubArray{<:Any, <:Any, <:MtlArray}) = a

# Metal only supports Float32
function on_architecture(::MetalGPU, s::StepRangeLen)
    ref = convert(Float32, s.ref)
    step = convert(Float32, s.step)
    len = s.len
    offset = s.offset
    return StepRangeLen(ref, step, len, offset)
end

@inline convert_to_device(::MetalGPU, args) = Metal.mtlconvert(args)
@inline convert_to_device(::MetalGPU, args::Tuple) = map(Metal.mtlconvert, args)

Metal.@device_override @inline function __validindex(ctx::MappedCompilerMetadata)
    if __dynamic_checkbounds(ctx)
        I = @inbounds linear_expand(__iterspace(ctx), threadgroup_position_in_grid_1d(),
                                thread_position_in_threadgroup_1d())
        return I in __linear_ndrange(ctx)
    else
        return true
    end
end

end # module

