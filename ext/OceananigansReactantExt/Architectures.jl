module Architectures

using Reactant
using Oceananigans

import Oceananigans.Architectures: device, architecture, array_type, on_architecture, unified_array, ReactantState, device_copy_to!

architecture(::Reactant.AnyConcreteRArray) = ReactantState
architecture(::Reactant.AnyTracedRArray) = ReactantState

array_type(::ReactantState) = ConcreteRArray

on_architecture(::ReactantState, a::Array) = ConcreteRArray(a)
on_architecture(::ReactantState, a::Reactant.AnyConcreteRArray) = a
on_architecture(::ReactantState, a::Reactant.AnyTracedRArray) = a
on_architecture(::ReactantState, a::BitArray) = ConcreteRArray(a)
on_architecture(::ReactantState, a::SubArray{<:Any, <:Any, <:Array}) = ConcreteRArray(a)

unified_array(::ReactantState, a) = a

@inline device_copy_to!(dst::Reactant.AnyConcreteRArray, src::Reactant.AnyConcreteRArray; kw...) = Base.copyto!(dst, src)

end # module
