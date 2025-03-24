module Architectures

using Reactant
using Oceananigans

using Reactant: AnyConcreteRArray

import Oceananigans.Architectures: device, architecture, array_type, on_architecture
import Oceananigans.Architectures: unified_array, ReactantState, device_copy_to!

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

const ReactantBackend = ReactantKernelAbstractionsExt.ReactantBackend

const AnyConcreteReactantArray = Union{Reactant.AnyConcretePJRTArray, Reactant.AnyConcreteIFRTArray}

device(::ReactantState) = ReactantBackend()

architecture(::AnyConcreteReactantArray) = ReactantState
architecture(::Reactant.AnyTracedRArray) = ReactantState

# ConcreteRArray can refer to either a PJRT or IFRT array based on Reactant preferences
array_type(::ReactantState) = ConcreteRArray

to_reactant_sharding(::Nothing) = Sharding.NoSharding()
to_reactant_sharding(s::Sharding.AbstractSharding) = s
to_reactant_sharding(::T) where {T} = error("Unsupported sharding type $T")

on_architecture(::ReactantState, a::Reactant.AnyTracedRArray) = a
on_architecture(::CPU, a::AnyConcreteReactantArray) = Array(a)
on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:AnyConcreteReactantArray}) = Array(a)

const ArraysToRArray = Union{Array,
                             Reactant.AnyConcretePJRTArray,
                             BitArray,
                             SubArray{<:Any, <:Any, <:Array}}

on_architecture(r::ReactantState, a::ArraysToRArray) =
    Reactant.to_rarray(a; sharding=to_reactant_sharding(r.sharding))

unified_array(::ReactantState, a) = a

@inline device_copy_to!(dst::AnyConcreteReactantArray, src::AnyConcreteReactantArray; kw...) = Base.copyto!(dst, src)

end # module
