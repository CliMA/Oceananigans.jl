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

"""
$(TYPEDSIGNATURES)

Return a `NamedTuple` with keys `names` and values `f(name)`. The values are built by
recursing over `names`, so that each `name` is a compile-time constant when `names` is;
`f` should start with `Base.@constprop :aggressive` when its result type depends on `name`.
The recursion does not go through `map` or `ntuple`, so that calls to those functions
inside `f` are not mistaken for recursion by the compiler.
"""
@inline named_tuple(f, names::Tuple) = NamedTuple{names}(map_names(f, names))

@inline map_names(f, ::Tuple{}) = ()
@inline map_names(f, names::Tuple) = (f(first(names)), map_names(f, Base.tail(names))...)

"""
$(TYPEDSIGNATURES)

Call `f(Val(n), Val(name))` for each `name` in `names`, where `n` is the position of `name`.
The recursion goes over the splatted `Val(name)`s rather than over `Val(names)`: the compiler
widens a recursion over `Val`-wrapped tuples of names from its third level on, which boxes
its arguments and dispatches the remaining calls dynamically.
"""
@inline foreach_name(f::F, ::Val{names}) where {F, names} = _foreach_name(f, Val(1), map(Val, names)...)

"""
$(TYPEDSIGNATURES)

Call `f(Val(n), Val(name))` for each property `name` of `nt`, where `n` is the position of `name`.
"""
@inline foreach_name(f::F, nt::NamedTuple) where F = foreach_name(f, Val(propertynames(nt)))

@inline _foreach_name(f, ::Val) = nothing

@inline function _foreach_name(f::F, ::Val{n}, val_name, val_names...) where {F, n}
    f(Val(n), val_name)
    return _foreach_name(f, Val(n + 1), val_names...)
end

@inline datatuple(obj::Nothing) = nothing
@inline datatuple(obj::AbstractArray) = obj
@inline datatuple(obj::Tuple) = Tuple(datatuple(o) for o in obj)
@inline datatuple(obj::NamedTuple) = NamedTuple{propertynames(obj)}(datatuple(o) for o in obj)
@inline datatuples(objs...) = (datatuple(obj) for obj in objs)

macro constprop(setting)
    if isa(setting, QuoteNode)
        setting = setting.value
    end
    setting === :aggressive && return Expr(:meta, :aggressive_constprop)
    setting === :none && return Expr(:meta, :no_constprop)
    throw(ArgumentError("@constprop $setting not supported"))
end
