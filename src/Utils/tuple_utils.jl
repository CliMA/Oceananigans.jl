import Oceananigans: tupleit

#####
##### Some utilities for tupling
#####

tupleit(::Nothing) = ()
tupleit(t::NamedTuple) = t
tupleit(t::Tuple) = t
tupleit(nt) = tuple(nt)
tupleit(nt::Vector) = tuple(nt...)

parenttuple(obj) = Tuple(f.data.parent for f in obj)

@inline datatuple(obj::Nothing) = nothing
@inline datatuple(obj::AbstractArray) = obj
@inline datatuple(obj::Tuple) = Tuple(datatuple(o) for o in obj)
@inline datatuple(obj::NamedTuple) = NamedTuple{propertynames(obj)}(datatuple(o) for o in obj)
@inline datatuples(objs...) = (datatuple(obj) for obj in objs)
