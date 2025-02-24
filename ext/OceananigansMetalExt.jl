module OceananigansMetalExt

using Metal

using Oceananigans
import Oceananigans.Architectures:
    architecture,
    convert_to_device,
    on_architecture

const MetalGPU = GPU{<:Metal.MetalBackend}
MetalGPU() = GPU(Metal.MetalBackend())

# elseif Sys.isapple() # Assumption!
#     return MetalGPU()
# end

architecture(::MtlArray) = MetalGPU()
on_architecture(::MetalGPU, a::Array) = MtlArray(a)
on_architecture(::MetalGPU, a::BitArray) = MtlArray(a)
on_architecture(::MetalGPU, a::SubArray{<:Any, <:Any, <:Array}) = MtlArray(a)
on_architecture(::CPU, a::MtlArray) = Array(a)
on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:MtlArray}) = Array(a)
on_architecture(::MetalGPU, a::MtlArray) = a
on_architecture(::MetalGPU, a::SubArray{<:Any, <:Any, <:MtlArray}) = a

@inline convert_to_device(::MetalGPU, args) = Metal.mtlconvert(args)
@inline convert_to_device(::MetalGPU, args::Tuple) = map(Metal.mtlconvert, args)

end # module
