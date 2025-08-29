using OffsetArrays
using Oceananigans.Grids: AbstractGrid

import Oceananigans.Architectures: on_architecture
import KernelAbstractions as KA
import Base: length


#####
##### Multi Region Object
#####

struct MultiRegionObject{R}
    regional_objects :: R
end

#####
##### Convenience structs
#####

struct Reference{R}
    ref :: R
end

struct Iterate{I}
    iter :: I
end

#####
##### Multi region functions
#####

@inline getregion(a, i) = a
@inline getregion(ref::Reference, i)        = ref.ref
@inline getregion(iter::Iterate, i)         = iter.iter[i]
@inline getregion(mo::MultiRegionObject, i) = _getregion(mo.regional_objects[i], i)
@inline getregion(p::Pair, i)               = p.first => _getregion(p.second, i)

@inline _getregion(a, i) = a
@inline _getregion(ref::Reference, i)        = ref.ref
@inline _getregion(iter::Iterate, i)         = iter.iter[i]
@inline _getregion(mo::MultiRegionObject, i) = getregion(mo.regional_objects[i], i)
@inline _getregion(p::Pair, i)               = p.first => getregion(p.second, i)

## The implementation of `getregion` for a Tuple forces the compiler to infer the size of the Tuple
@inline getregion(t::Tuple{}, i)                     = ()
@inline getregion(t::Tuple{<:Any}, i)                = (_getregion(t[1], i), )
@inline getregion(t::Tuple{<:Any, <:Any}, i)         = (_getregion(t[1], i), _getregion(t[2], i))
@inline getregion(t::Tuple{<:Any, <:Any, <:Any}, i)  = (_getregion(t[1], i), _getregion(t[2], i), _getregion(t[3], i))
@inline getregion(t::Tuple, i)                       = (_getregion(t[1], i), _getregion(t[2:end], i)...)

@inline _getregion(t::Tuple{}, i)                    = ()
@inline _getregion(t::Tuple{<:Any}, i)               = (getregion(t[1], i), )
@inline _getregion(t::Tuple{<:Any, <:Any}, i)        = (getregion(t[1], i), getregion(t[2], i))
@inline _getregion(t::Tuple{<:Any, <:Any, <:Any}, i) = (getregion(t[1], i), getregion(t[2], i), getregion(t[3], i))
@inline _getregion(t::Tuple, i)                      = (getregion(t[1], i), getregion(t[2:end], i)...)

@inline getregion(nt::NamedTuple, i)  = NamedTuple{keys(nt)}(_getregion(Tuple(nt), i))
@inline _getregion(nt::NamedTuple, i) = NamedTuple{keys(nt)}(getregion(Tuple(nt), i))

@inline isregional(a)                   = false
@inline isregional(::MultiRegionObject) = true

@inline isregional(t::Tuple{}) = false
@inline isregional(nt::NT) where NT<:NamedTuple{(), Tuple{}} = false
for func in [:isregional, :regions, :switch_device!]
    @eval begin
        @inline $func(t::Union{Tuple, NamedTuple}) = $func(first(t))
    end
end

@inline regions(mo::MultiRegionObject) = 1:length(mo.regional_objects)

Base.getindex(mo::MultiRegionObject, i, args...) = Base.getindex(mo.regional_objects, i, args...)
Base.length(mo::MultiRegionObject)               = Base.length(mo.regional_objects)

Base.similar(mo::MultiRegionObject) = construct_regionally(similar, mo)
Base.parent(mo::MultiRegionObject) = construct_regionally(parent, mo)

on_architecture(arch, mo::MultiRegionObject) = MultiRegionObject(on_architecture(arch, mo.regional_objects))

# For non-returning functions -> can we make it NON BLOCKING? This seems to be synchronous!
@inline function apply_regionally!(regional_func!, args...; kwargs...)
    multi_region_args   = isnothing(findfirst(isregional, args))   ? nothing : args[findfirst(isregional, args)]
    multi_region_kwargs = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
    isnothing(multi_region_args) && isnothing(multi_region_kwargs) && return regional_func!(args...; kwargs...)

    R = isnothing(multi_region_args) ? regions(multi_region_kwargs) : regions(multi_region_args)

    for r in R
        switch_device!(dev)
        regional_func!((getregion(arg, r) for arg in args)...; (getregion(kwarg, r) for kwarg in kwargs)...)
    end

    return nothing
end

@inline construct_regionally(regional_func::Base.Callable, args...; kwargs...) =
    construct_regionally(1, regional_func, args...; kwargs...)

# For functions with return statements -> BLOCKING! (use as seldom as possible)
@inline function construct_regionally(Nreturns::Int, regional_func::Base.Callable, args...; kwargs...)
    # First, we deduce whether any of `args` or `kwargs` are multi-regional.
    # If no regional objects are found, we call the function as usual
    multi_region_args   = isnothing(findfirst(isregional, args))   ? nothing :   args[findfirst(isregional, args)]
    multi_region_kwargs = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
    isnothing(multi_region_args) && isnothing(multi_region_kwargs) && return regional_func(args...; kwargs...)

    R = isnothing(multi_region_args) ? regions(multi_region_kwargs) : regions(multi_region_args)

    # Evaluate regional_func on the device of that region and collect
    # return values
    regional_return_values = Vector(undef, length(R))
    for r in R
        regional_return_values[r] = regional_func((getregion(arg, r) for arg in args)...;
                                                  (getregion(kwarg, r) for kwarg in kwargs)...)
    end

    if Nreturns == 1
        return MultiRegionObject(Tuple(regional_return_values))
    else
        return Tuple(MultiRegionObject(Tuple(regional_return_values[r][i] for r in 1:length(R))) for i in 1:Nreturns)
    end
end

@inline sync_device!(::Nothing)  = nothing
@inline sync_device!(::CPU)      = nothing

# TODO: The macro errors when there is a return and the function has (args...) in the
# signature (example using a macro on `multi_region_boundary_conditions:L74)

"""
    @apply_regionally expr

Distributes locally the function calls in `expr`ession

When the function call in `expr` does not return anything, then `apply_regionally!` method is used.
When the function in `expr` returns something, the `construct_regionally` method is used.
"""
macro apply_regionally(expr)
    if expr.head == :call
        func = expr.args[1]
        args = expr.args[2:end]
        multi_region = quote
            $(apply_regionally!)($func, $(args...))
        end
        return quote
            $(esc(multi_region))
        end
    elseif expr.head == :(=)
        ret = expr.args[1]
        Nret = 1
        if expr.args[1] isa Expr
            Nret = length(expr.args[1].args)
        end
        exp = expr.args[2]
        if exp isa Symbol # It is not a function call! Just a variable assignment
            return quote
                $ret = $(esc(exp))
            end
        end
        func = exp.args[1]
        args = exp.args[2:end]
        multi_region = quote
            $ret = $(construct_regionally)($Nret, $func, $(args...))
        end
        return quote
            $(esc(multi_region))
        end
    elseif expr.head == :block
        new_expr = deepcopy(expr)
        for (idx, arg) in enumerate(expr.args)
            if arg isa Expr && arg.head == :call
                func = arg.args[1]
                args = arg.args[2:end]
                new_expr.args[idx] = quote
                    $(apply_regionally!)($func, $(args...))
                end
            elseif arg isa Expr && arg.head == :(=)
                ret = arg.args[1]
                Nret = 1
                if arg.args[1] isa Expr
                    Nret = length(arg.args[1].args)
                end
                exp = arg.args[2]
                func = exp.args[1]
                args = exp.args[2:end]
                new_expr.args[idx] = quote
                    $ret = $(construct_regionally)($Nret, $func, $(args...))
                end
            end
        end
        return quote
            $(esc(new_expr))
        end
    end
end
