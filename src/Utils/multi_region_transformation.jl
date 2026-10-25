using Oceananigans.Architectures: Architectures, on_architecture, CPU

import Oceananigans: prognostic_state, restore_prognostic_state!

import KernelAbstractions as KA

#####
##### Multi Region Object
#####

struct MultiRegionObject{R}
    regional_objects :: R
end

Base.eltype(mo::MultiRegionObject) = eltype(first(mo.regional_objects))

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

@inline isregional(::Tuple{}) = false
@inline isregional(t::Tuple) = isregional(first(t)) || isregional(Base.tail(t))
@inline isregional(nt::NamedTuple) = isregional(values(nt))

@inline regions(t::Union{Tuple, NamedTuple}) = regions(first(t))
@inline regions(mo::MultiRegionObject) = 1:length(mo.regional_objects)

Base.getindex(mo::MultiRegionObject, i, args...) = Base.getindex(mo.regional_objects, i, args...)
Base.length(mo::MultiRegionObject)               = Base.length(mo.regional_objects)

Base.similar(mo::MultiRegionObject) = construct_regionally(similar, mo)
Base.parent(mo::MultiRegionObject) = construct_regionally(parent, mo)

Architectures.on_architecture(arch, mo::MultiRegionObject) = MultiRegionObject(on_architecture(arch, mo.regional_objects))

# `isregional(args)` and `isregional(values(kwargs))` fold to compile-time constants for
# non-regional arguments, so the multi-region branch is dead code (and is not inferred) for them.
# The `::F` and `Vararg{Any, N}` annotations make the compiler specialize on arguments that are only passed on.
@inline function apply_regionally!(regional_func!::F, args::Vararg{Any, N}; kwargs...) where {F, N}
    if isregional(args) || isregional(values(kwargs))
        return multi_region_apply!(regional_func!, args...; kwargs...)
    else
        return regional_func!(args...; kwargs...)
    end
end

# The region index is a type parameter so that indexing heterogeneous region tuples
# (e.g. cubed sphere panels whose rotations have different types) is type-stable.
struct RegionSelector{R} end

@inline (::RegionSelector{R})(a) where R = getregion(a, R)

@inline first_regional(::Tuple{}) = nothing
@inline first_regional(t::Tuple) = isregional(first(t)) ? first(t) : first_regional(Base.tail(t))

@inline regional_function(regional_func!, selector) = regional_func!
@inline regional_function(regional_func!::MultiRegionObject, selector) = selector(regional_func!)

@inline function apply_in_region!(regional_func!::F, selector, args, kwargs) where F
    regional_function(regional_func!, selector)(map(selector, args)...; map(selector, kwargs)...)
    return nothing
end

# Generated rather than recursive in Val{r + 1}: inference widens a self-call whose type signature grows, which
# turned each region step into a dynamic dispatch.
@generated function apply_in_regions!(regional_func!, ::Val{N}, args, kwargs) where N
    calls = [:(apply_in_region!(regional_func!, RegionSelector{$r}(), args, kwargs)) for r in 1:N]
    return quote
        $(calls...)
        return nothing
    end
end

# Any iterable of regions (e.g. a custom `regions` sequence) keeps its own identifiers and order.
@inline function apply_in_each_region!(regional_func!::F, R, args, kwargs) where F
    for r in R
        selector = Base.Fix2(getregion, r)
        regional_function(regional_func!, selector)(map(selector, args)...; map(selector, kwargs)...)
    end
    return nothing
end

@inline apply_in_regions!(regional_func!::F, R, args, kwargs) where F = apply_in_each_region!(regional_func!, R, args, kwargs)

@inline function apply_in_regions!(regional_func!::F, R::AbstractUnitRange{<:Integer}, args, kwargs) where F
    if first(R) == 1
        apply_in_regions!(regional_func!, Val(length(R)), args, kwargs)
    else
        apply_in_each_region!(regional_func!, R, args, kwargs)
    end
    return nothing
end

# A region count known from the type (the default `1:N` regions of a tuple of N regional objects) lets the region loop
# unroll with no dynamic dispatch at its entry; anything else falls back to `regions`.
@inline static_regions(a) = regions(a)
@inline static_regions(mo::MultiRegionObject{<:NTuple{N, Any}}) where N = Val(N)

@inline function multi_region_apply!(regional_func!::F, args::Vararg{Any, N}; kwargs...) where {F, N}
    regional_argument = first_regional(args)
    regional_argument = isnothing(regional_argument) ? first_regional(Tuple(values(kwargs))) : regional_argument
    apply_in_regions!(regional_func!, static_regions(regional_argument), args, values(kwargs))
    return nothing
end

# For `MultiRegionObject` calls (think kernels with different sizes for different regions)
# we also `getregion` for the function. Since the function itself is multiregional we do not
# need to check if the call is single region or not.
@inline function apply_regionally!(regional_func!::MultiRegionObject, args::Vararg{Any, N}; kwargs...) where N
    apply_in_regions!(regional_func!, static_regions(regional_func!), args, values(kwargs))
    return nothing
end

@inline construct_regionally(regional_func::Base.Callable, args...; kwargs...) =
    construct_regionally(1, regional_func, args...; kwargs...)

# For functions with return statements -> BLOCKING! (use as seldom as possible)
@inline function construct_regionally(Nreturns::Int, regional_func::Base.Callable, args...; kwargs...)
    if isregional(args) || isregional(values(kwargs))
        return multi_region_construct(Nreturns, regional_func, args...; kwargs...)
    else
        return regional_func(args...; kwargs...)
    end
end

@inline function multi_region_construct(Nreturns::Int, regional_func::Base.Callable, args...; kwargs...)
    multi_region_args   = isnothing(findfirst(isregional, args))   ? nothing :   args[findfirst(isregional, args)]
    multi_region_kwargs = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]

    R = isnothing(multi_region_args) ? regions(multi_region_kwargs) : regions(multi_region_args)

    # Evaluate regional_func on the device of that region and collect
    # return values
    regional_return_values = Vector(undef, length(R))
    for r in R
        regional_return_values[r] = regional_func(map(Base.Fix2(getregion, r), args)...; map(Base.Fix2(getregion, r), values(kwargs))...)
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

#####
##### Checkpointing
#####

function prognostic_state(mo::MultiRegionObject)
    return Tuple(prognostic_state(regional_obj) for regional_obj in mo.regional_objects)
end

function restore_prognostic_state!(restored::MultiRegionObject, from)
    regional_states = from isa MultiRegionObject ? from.regional_objects : from

    for (regional_obj, regional_state) in zip(restored.regional_objects, regional_states)
        restore_prognostic_state!(regional_obj, regional_state)
    end
    return restored
end

restore_prognostic_state!(::MultiRegionObject, ::Nothing) = nothing
