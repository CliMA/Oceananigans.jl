module OceananigansoneAPIExt

using oneAPI
using Oceananigans

import Oceananigans.Architectures:
    architecture,
    convert_to_device,
    on_architecture

# maybe it is more consistent to call this oneAPIGPU, but that is hard to read and they do oneArray so maybe this is fine?
const oneGPU = Oceananigans.GPU{<:oneAPI.oneAPIBackend}
oneGPU() = Oceananigans.GPU(oneAPI.oneAPIBackend())

architecture(::oneGPU) = oneGPU()

on_architecture(::oneGPU, a::Number) = a
on_architecture(::oneGPU, a::Array) = oneArray(a)
on_architecture(::oneGPU, a::BitArray) = oneArray(a)
on_architecture(::oneGPU, a::SubArray{<:Any, <:Any, <:Array}) = oneArray(a)
on_architecture(::CPU, a::oneArray) = Array(a)
on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:oneArray}) = Array(a)
on_architecture(::oneGPU, a::oneArray) = a
on_architecture(::oneGPU, a::SubArray{<:Any, <:Any, <:oneArray}) = a

#= Some oneAPI GPUs don't support Float32 so we might need something here
# Metal only supports Float32 
function on_architecture(::oneGPU, s::StepRangeLen)
    ref = convert(Float32, s.ref)
    step = convert(Float32, s.step)
    len = s.len
    offset = s.offset
    return StepRangeLen(ref, step, len, offset)
end
=#
#=idk what these do
@inline convert_to_device(::oneGPU, args) = Metal.mtlconvert(args)
@inline convert_to_device(::oneGPU, args::Tuple) = map(Metal.mtlconvert, args)
=#
end # module
