module Architectures

using Reactant
using Oceananigans

import Oceananigans.Architectures: device, architecture, array_type, on_architecture, unified_array, ReactantState, device_copy_to!

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)
const ReactantBackend = ReactantKernelAbstractionsExt.ReactantBackend
device(::ReactantState) = ReactantBackend()

architecture(::Reactant.AnyConcretePJRTArray) = ReactantState
architecture(::Reactant.AnyTracedRArray) = ReactantState

array_type(::ReactantState) = ConcretePJRTArray

to_reactant_sharding(::Missing) = Sharding.NoSharding()
to_reactant_sharding(s::Sharding.AbstractSharding) = s
to_reactant_sharding(::T) where {T} = error("Unsupported sharding type $T")

on_architecture(::ReactantState, a::Reactant.AnyTracedRArray) = a
function on_architecture(r::ReactantState, a::Array)
    return Reactant.to_rarray(a; sharding=to_reactant_sharding(r.sharding))
end
function on_architecture(r::ReactantState, a::Reactant.AnyConcretePJRTArray)
    return Reactant.to_rarray(a; sharding=to_reactant_sharding(r.sharding))
end
function on_architecture(r::ReactantState, a::BitArray)
    return Reactant.to_rarray(a; sharding=to_reactant_sharding(r.sharding))
end
function on_architecture(r::ReactantState, a::SubArray{<:Any,<:Any,<:Array})
    return Reactant.to_rarray(a; sharding=to_reactant_sharding(r.sharding))
end

unified_array(::ReactantState, a) = a

@inline device_copy_to!(dst::Reactant.AnyConcretePJRTArray, src::Reactant.AnyConcretePJRTArray; kw...) = Base.copyto!(dst, src)

end # module
