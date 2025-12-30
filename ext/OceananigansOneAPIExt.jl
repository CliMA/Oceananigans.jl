module OceananigansOneAPIExt

using oneAPI
using Oceananigans

import Oceananigans.Architectures:
    architecture,
    convert_to_device,
    on_architecture

const ONEGPU = GPU{<:oneAPI.oneAPIBackend}
ONEGPU() = GPU(oneAPI.oneAPIBackend())

architecture(::oneArray) = ONEGPU()
Base.summary(::ONEGPU) = "ONEGPU"

on_architecture(::ONEGPU, a::Number) = a
on_architecture(::ONEGPU, a::Array) = oneArray(a)
on_architecture(::ONEGPU, a::BitArray) = oneArray(a)
on_architecture(::ONEGPU, a::SubArray{<:Any, <:Any, <:Array}) = oneArray(a)
on_architecture(::CPU, a::oneArray) = Array(a)
on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:oneArray}) = Array(a)
on_architecture(::ONEGPU, a::oneArray) = a
on_architecture(::ONEGPU, a::SubArray{<:Any, <:Any, <:oneArray}) = a
on_architecture(::ONEGPU, a::StepRangeLen) = a

@inline convert_to_device(::ONEGPU, args) = oneAPI.kernel_convert(args)
@inline convert_to_device(::ONEGPU, args::Tuple) = map(oneAPI.kernel_convert, args)

end # module
