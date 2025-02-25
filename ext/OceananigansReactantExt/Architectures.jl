module Architectures

using Reactant
using Oceananigans

import Oceananigans.Architectures: device, architecture, array_type, on_architecture, unified_array, ReactantState, device_copy_to!

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)
const ReactantBackend = ReactantKernelAbstractionsExt.ReactantBackend
device(::ReactantState) = ReactantBackend()

architecture(::Reactant.AnyConcreteRArray) = ReactantState
architecture(::Reactant.AnyTracedRArray) = ReactantState

array_type(::ReactantState) = ConcreteRArray

to_reactant_sharding(::Missing) = Sharding.NoSharding()
to_reactant_sharding(s::Sharding.AbstractSharding) = s
to_reactant_sharding(::T) where {T} = error("Unsupported sharding type $T")

on_architecture(::ReactantState, a::Reactant.AnyTracedRArray) = a
function on_architecture(r::ReactantState, a::Array)
    # XXX: Only for testing purposes
    sharding = ndims(a) == 2 ? to_reactant_sharding(r.sharding) : Sharding.NoSharding()
    return ConcreteRArray(a; sharding)
end
function on_architecture(r::ReactantState, a::Reactant.AnyConcreteRArray)
    # XXX: Only for testing purposes
    sharding = ndims(a) == 2 ? to_reactant_sharding(r.sharding) : Sharding.NoSharding()
    return ConcreteRArray(a; sharding)
end
function on_architecture(r::ReactantState, a::BitArray)
    # XXX: Only for testing purposes
    sharding = ndims(a) == 2 ? to_reactant_sharding(r.sharding) : Sharding.NoSharding()
    return ConcreteRArray(a; sharding)
end
function on_architecture(r::ReactantState, a::SubArray{<:Any,<:Any,<:Array})
    # XXX: Only for testing purposes
    sharding = ndims(a) == 2 ? to_reactant_sharding(r.sharding) : Sharding.NoSharding()
    return ConcreteRArray(a; sharding)
end

function Base.zeros(arch::ReactantState, FT, N...)
    # XXX: Only for testing purposes
    N == 2 && return on_architecture(arch, zeros(FT, N...))
    return ConcreteRArray(zeros(FT, N...))
end

unified_array(::ReactantState, a) = a

@inline device_copy_to!(dst::Reactant.AnyConcreteRArray, src::Reactant.AnyConcreteRArray; kw...) = Base.copyto!(dst, src)

end # module
