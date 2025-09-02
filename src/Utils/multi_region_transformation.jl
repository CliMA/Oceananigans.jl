using OffsetArrays
using Oceananigans.Grids: AbstractGrid

import Oceananigans.Architectures: on_architecture
import KernelAbstractions as KA
import Base: length


#####
##### Multi Region Object
#####

struct MultiRegionObject{R, D, B}
    regional_objects :: R
    devices :: D
    backend :: B

    function MultiRegionObject(backend::KA.Backend, regional_objects...; devices=Tuple(CPU() for _ in regional_objects))
        R = typeof(regional_objects)
        D = typeof(devices)
        B = typeof(backend)
        return new{R, D, B}(regional_objects, devices, backend)
    end

    function MultiRegionObject(backend::KA.Backend, regional_objects::Tuple, devices::Tuple)
        R = typeof(regional_objects)
        D = typeof(devices)
        B = typeof(backend)
        return new{R, D, B}(regional_objects, devices, backend)
    end
end

MultiRegionObject(arch::AbstractArchitecture, regional_objects...; devices=Tuple(CPU() for _ in regional_objects)) =
    MultiRegionObject(device(arch), regional_objects...; devices=devices)
MultiRegionObject(arch::AbstractArchitecture, regional_objects::Tuple, devices::Tuple) =
    MultiRegionObject(device(arch), regional_objects, devices)

"""
    MultiRegionObject(arch::AbstractArchitecture, regional_objects::Tuple; devices)

Return a MultiRegionObject
"""
MultiRegionObject(arch::AbstractArchitecture, regional_objects::Tuple; devices=Tuple(CPU() for _ in regional_objects)) =
    MultiRegionObject(arch, regional_objects, devices)


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
@inline getbackend(mo::MultiRegionObject) = mo.backend
@inline getdevice(a, i)                     = nothing
@inline getdevice(cu::OffsetArray, i)       = getdevice(cu.parent)
@inline getdevice(mo::MultiRegionObject, i) = mo.devices[i]

@inline getdevice(a)               = nothing
@inline getdevice(cu::OffsetArray) = getdevice(cu.parent)

@inline switch_device!(a)                        = nothing
@inline switch_device!(dev::Int)                 = device!(dev)
@inline switch_device!(dev::Tuple, i)            = switch_device!(dev[i])
@inline switch_device!(mo::MultiRegionObject, i) = switch_device!(getdevice(mo, i))

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
for func in [:isregional, :devices, :switch_device!]
    @eval begin
        @inline $func(t::Union{Tuple, NamedTuple}) = $func(first(t))
    end
end

@inline devices(mo::MultiRegionObject) = mo.devices

Base.getindex(mo::MultiRegionObject, i, args...) = Base.getindex(mo.regional_objects, i, args...)
Base.length(mo::MultiRegionObject)               = Base.length(mo.regional_objects)

Base.similar(mo::MultiRegionObject) = construct_regionally(similar, mo)
Base.parent(mo::MultiRegionObject) = construct_regionally(parent, mo)

on_architecture(arch::CPU, mo::MultiRegionObject) = MultiRegionObject(arch, on_architecture(arch, mo.regional_objects))

# TODO: Properly define on_architecture(::GPU, mo::MultiRegionObject) to handle cases where MultiRegionObject can be
# distributed across different devices. Currently, the implementation assumes that all regional objects reside on a
# single GPU.
on_architecture(arch::GPU, mo::MultiRegionObject) =
    MultiRegionObject(arch, on_architecture(arch, mo.regional_objects);
                      devices = Tuple(device(arch) for i in 1:length(mo.regional_objects)))

# For non-returning functions -> can we make it NON BLOCKING? This seems to be synchronous!
@inline function apply_regionally!(regional_func!, args...; kwargs...)
    multi_region_args   = isnothing(findfirst(isregional, args))   ? nothing : args[findfirst(isregional, args)]
    multi_region_kwargs = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
    isnothing(multi_region_args) && isnothing(multi_region_kwargs) && return regional_func!(args...; kwargs...)

    devs = isnothing(multi_region_args) ? multi_region_kwargs : multi_region_args
    devs = devices(devs)

    for (r, dev) in enumerate(devs)
        switch_device!(dev)
        regional_func!((getregion(arg, r) for arg in args)...; (getregion(kwarg, r) for kwarg in kwargs)...)
    end

    sync_all_devices!(devs)

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

    devs = isnothing(multi_region_args) ? multi_region_kwargs : multi_region_args
    devs = devices(devs)

    # Dig out the backend since we don't have access to arch.
    backend = nothing
    for arg in args
        if arg isa MultiRegionObject
            backend = getbackend(arg)
            break
        end
    end

    if backend isa Nothing
        backend = devs[1]
    end

    # Evaluate regional_func on the device of that region and collect
    # return values
    regional_return_values = Vector(undef, length(devs))
    for (r, dev) in enumerate(devs)
        switch_device!(dev)
        regional_return_values[r] = regional_func((getregion(arg, r) for arg in args)...;
                                                  (getregion(kwarg, r) for kwarg in kwargs)...)
    end
    sync_all_devices!(devs)

    if Nreturns == 1
        return MultiRegionObject(backend, Tuple(regional_return_values), devs)
    else
        return Tuple(MultiRegionObject(backend, Tuple(regional_return_values[r][i] for r in 1:length(devs)), devs) for i in 1:Nreturns)
    end
end

@inline sync_all_devices!(grid::AbstractGrid)    = nothing
@inline sync_all_devices!(mo::MultiRegionObject) = sync_all_devices!(devices(mo))

@inline function sync_all_devices!(devices)
    for dev in devices
        switch_device!(dev)
        sync_device!(dev)
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
