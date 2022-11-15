using CUDA: CuArray, CuDevice, CuContext, CuPtr, device, device!, synchronize
using OffsetArrays
using Oceananigans.Grids: AbstractGrid
import Base: length

const GPUVar = Union{CuArray, CuContext, CuPtr, Ptr}

### 
### Multi Region Object
###

struct MultiRegionObject{R, D}
    regions :: R
    devices :: D
end

###
### Convenience structs 
###

struct Reference{R}
    ref :: R
end

struct Iterate{I}
    iter :: I
end

###
### Multi region functions
###

@inline getdevice(a, i)                     = nothing
@inline getdevice(cu::GPUVar, i)            = CUDA.device(cu)
@inline getdevice(cu::OffsetArray, i)       = getdevice(cu.parent)
@inline getdevice(mo::MultiRegionObject, i) = mo.devices[i]

@inline getdevice(a)               = nothing
@inline getdevice(cu::GPUVar)      = CUDA.device(cu)
@inline getdevice(cu::OffsetArray) = getdevice(cu.parent)

@inline switch_device!(a)                        = nothing
@inline switch_device!(dev::Int)                 = CUDA.device!(dev)
@inline switch_device!(dev::CuDevice)            = CUDA.device!(dev)
@inline switch_device!(dev::Tuple, i)            = switch_device!(dev[i])
@inline switch_device!(mo::MultiRegionObject, i) = switch_device!(getdevice(mo, i))

@inline getregion(a, i) = a
@inline getregion(ref::Reference, i)        = ref.ref
@inline getregion(iter::Iterate, i)         = iter.iter[i]
@inline getregion(mo::MultiRegionObject, i) = _getregion(mo.regions[i], i)
@inline getregion(p::Pair, i)               = p.first => _getregion(p.second, i)

@inline _getregion(a, i) = a
@inline _getregion(ref::Reference, i)        = ref.ref
@inline _getregion(iter::Iterate, i)         = iter.iter[i]
@inline _getregion(mo::MultiRegionObject, i) = getregion(mo.regions[i], i)
@inline _getregion(p::Pair, i)               = p.first => getregion(p.second, i)

## The implementation of `getregion` for a Tuple forces the compiler to infer the size of the Tuple
@inline getregion(t::Tuple{}, i)                            = ()
@inline getregion(t::Tuple{<:Any}, i)                       = (_getregion(t[1], i), )
@inline getregion(t::Tuple{<:Any, <:Any}, i)                = (_getregion(t[1], i), _getregion(t[2], i))
@inline getregion(t::Tuple{<:Any, <:Any, <:Any}, i)         = (_getregion(t[1], i), _getregion(t[2], i), _getregion(t[3], i))
@inline getregion(t::Tuple, i)                              = (_getregion(t[1], i), _getregion(t[2:end], i)...)

@inline _getregion(t::Tuple{}, i)                           = ()
@inline _getregion(t::Tuple{<:Any}, i)                      = (getregion(t[1], i), )
@inline _getregion(t::Tuple{<:Any, <:Any}, i)               = (getregion(t[1], i), getregion(t[2], i))
@inline _getregion(t::Tuple{<:Any, <:Any, <:Any}, i)        = (getregion(t[1], i), getregion(t[2], i), getregion(t[3], i))
@inline _getregion(t::Tuple, i)                             = (getregion(t[1], i), getregion(t[2:end], i)...)

@inline getregion(nt::NamedTuple, i)   = NamedTuple{keys(nt)}(_getregion(Tuple(nt), i))
@inline _getregion(nt::NamedTuple, i)  = NamedTuple{keys(nt)}(getregion(Tuple(nt), i))

@inline isregional(a)                   = false
@inline isregional(::MultiRegionObject) = true

@inline isregional(t::Tuple{}) = false
@inline isregional(nt::NT) where NT<:NamedTuple{<:Any, Tuple{Tuple{}}} = false
for func in [:isregional, :devices, :switch_device!]
    @eval begin
        @inline $func(t::Union{Tuple, NamedTuple}) = $func(first(t))
    end
end

@inline devices(mo::MultiRegionObject) = mo.devices

Base.getindex(mo::MultiRegionObject, i, args...) = Base.getindex(mo.regions, i, args...)
Base.length(mo::MultiRegionObject)               = Base.length(mo.regions)

# For non-returning functions -> can we make it NON BLOCKING? This seems to be synchronous!
@inline function apply_regionally!(func!, args...; kwargs...)
    mra = isnothing(findfirst(isregional, args)) ? nothing : args[findfirst(isregional, args)]
    mrk = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
    isnothing(mra) && isnothing(mrk) && return func!(args...; kwargs...)

    if isnothing(mra) 
        devs = devices(mrk)
    else
        devs = devices(mra)
    end
   
    for (r, dev) in enumerate(devs)
        switch_device!(dev)
        func!((getregion(arg, r) for arg in args)...; (getregion(kwarg, r) for kwarg in kwargs)...)
    end

    sync_all_devices!(devs)
end 

# For functions with return statements -> BLOCKING! (use as seldom as possible)
@inline function construct_regionally(constructor, args...; kwargs...)
    mra = isnothing(findfirst(isregional, args)) ? nothing : args[findfirst(isregional, args)]
    mrk = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
    isnothing(mra) && isnothing(mrk) && return constructor(args...; kwargs...)

    if isnothing(mra) 
        devs = devices(mrk)
    else
        devs = devices(mra)
    end

    res = Vector(undef, length(devs))
    for (r, dev) in enumerate(devs)
        switch_device!(dev)
        res[r] = constructor((getregion(arg, r) for arg in args)...; (getregion(kwarg, r) for kwarg in kwargs)...)
    end
    sync_all_devices!(devs)

    return MultiRegionObject(Tuple(res), devs)
end

@inline sync_all_devices!(grid::AbstractGrid)    = nothing
@inline sync_all_devices!(mo::MultiRegionObject) = sync_all_devices!(devices(mo))

@inline function sync_all_devices!(devices)
    for dev in devices
        switch_device!(dev)
        sync_device!(dev)
    end
end

@inline sync_device!(::CuDevice) = CUDA.device_synchronize()
@inline sync_device!(dev)        = nothing

"""
    @apply_regionally expr
    
Use `@apply_regionally` to distribute locally the function calls.
Call `compute_regionally` in case of a returning value and `apply_regionally!` 
in case of no return.
"""
macro apply_regionally(expr)
    if expr.head == :call
        func = expr.args[1]
        args = expr.args[2:end]
        multi_region = quote
            apply_regionally!($func, $(args...))
        end
        return quote
            $(esc(multi_region))
        end
    elseif expr.head == :(=)
        ret = expr.args[1]
        exp = expr.args[2]
        func = exp.args[1]
        args = exp.args[2:end]
        multi_region = quote
            $ret = construct_regionally($func, $(args...))
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
                    apply_regionally!($func, $(args...))
                end
            elseif arg isa Expr && arg.head == :(=)
                ret = arg.args[1]
                exp = arg.args[2]
                func = exp.args[1]
                args = exp.args[2:end]
                new_expr.args[idx] = quote
                    $ret = construct_regionally($func, $(args...))
                end
            end
        end
        return quote
            $(esc(new_expr))
        end
    end
end
