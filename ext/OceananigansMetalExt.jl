module OceananigansMetalExt

using Metal
using Oceananigans

import Oceananigans.Architectures:
    architecture,
    convert_to_device,
    on_architecture

const MetalGPU = GPU{<:Metal.MetalBackend}
MetalGPU() = GPU(Metal.MetalBackend())

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

end # module

