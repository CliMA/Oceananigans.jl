module OceananigansOneAPIExt

using Oceananigans: Oceananigans, CPU, GPU
using oneAPI: oneAPI, oneArray

const ONEGPU = GPU{<:oneAPI.oneAPIBackend}
ONEGPU() = GPU(oneAPI.oneAPIBackend())

Oceananigans.Architectures.architecture(::oneArray) = ONEGPU()
Base.summary(::ONEGPU) = "ONEGPU"

Oceananigans.Architectures.on_architecture(::ONEGPU, a::Number) = a
Oceananigans.Architectures.on_architecture(::ONEGPU, a::Array) = oneArray(a)
Oceananigans.Architectures.on_architecture(::ONEGPU, a::BitArray) = oneArray(a)
Oceananigans.Architectures.on_architecture(::CPU,    a::oneArray) = Array(a)
Oceananigans.Architectures.on_architecture(::ONEGPU, a::oneArray) = a
Oceananigans.Architectures.on_architecture(::ONEGPU, a::StepRangeLen) = a

@inline Oceananigans.Architectures.convert_to_device(::ONEGPU, args) = oneAPI.kernel_convert(args)
@inline Oceananigans.Architectures.convert_to_device(::ONEGPU, args::Tuple) = map(oneAPI.kernel_convert, args)

end # module
