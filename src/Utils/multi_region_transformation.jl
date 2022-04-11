using CUDA: CuArray, CuDevice, CuContext, CuPtr, device, device!, synchronize
using OffsetArrays
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

getdevice(a, i)                     = nothing
getdevice(cu::GPUVar, i)            = CUDA.device(cu)
getdevice(cu::OffsetArray, i)       = getdevice(cu.parent)
getdevice(mo::MultiRegionObject, i) = mo.devices[i]

getdevice(a)               = nothing
getdevice(cu::GPUVar)      = CUDA.device(cu)
getdevice(cu::OffsetArray) = getdevice(cu.parent)

switch_device!(a)                        = nothing
switch_device!(dev::Int)                 = CUDA.device!(dev)
switch_device!(dev::CuDevice)            = CUDA.device!(dev)
switch_device!(dev::Tuple, i)            = switch_device!(dev[i])
switch_device!(mo::MultiRegionObject, i) = switch_device!(getdevice(mo, i))

@inline getregion(a, i) = a
@inline getregion(ref::Reference, i)        = ref.ref
@inline getregion(iter::Iterate, i)         = iter.iter[i]
@inline getregion(mo::MultiRegionObject, i) = getregion(mo.regions[i], i)

@inline getregion(t::Tuple, i)              = Tuple(getregion(elem, i) for elem in t)
@inline getregion(nt::NamedTuple, i)        = NamedTuple{keys(nt)}(getregion(elem, i) for elem in nt)

isregional(a)                   = false
isregional(::MultiRegionObject) = true
isregional(t::Union{Tuple, NamedTuple}) = any([isregional(elem) for elem in t])

for func in [:devices, :switch_device!]
    @eval begin
        $func(t::Union{Tuple, NamedTuple}) = $func(t[1])
    end
end

devices(mo::MultiRegionObject) = mo.devices

Base.getindex(mo::MultiRegionObject, i, args...) = Base.getindex(mo.regions, i, args...)
Base.length(mo::MultiRegionObject)               = Base.length(mo.regions)

# For non-returning functions -> can we make it NON BLOCKING? This seems to be synchronous!
function apply_regionally!(func!, args...; kwargs...)
    mra = isnothing(findfirst(isregional, args)) ? nothing : args[findfirst(isregional, args)]
    mrk = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
    isnothing(mra) && isnothing(mrk) && return func!(args...; kwargs...)

    if isnothing(mra) 
        devs = devices(mrk)
    else
        devs = devices(mra)
    end
    
    @sync for (r, dev) in enumerate(devs)
        @async begin
            switch_device!(dev)
            func!((getregion(arg, r) for arg in args)...; (getregion(kwarg, r) for kwarg in kwargs)...)
        end
    end

    sync_all_devices!(devs)
end 

# For functions with return statements -> BLOCKING! (use as seldom as possible)
function construct_regionally(constructor, args...; kwargs...)
    mra = isnothing(findfirst(isregional, args)) ? nothing : args[findfirst(isregional, args)]
    mrk = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
    isnothing(mra) && isnothing(mrk) && return constructor(args...; kwargs...)

    if isnothing(mra) 
        devs = devices(mrk)
    else
        devs = devices(mra)
    end

    res = Tuple((switch_device!(dev);
                constructor((getregion(arg, r) for arg in args)...; (getregion(kwarg, r) for kwarg in kwargs)...))
                for (r, dev) in enumerate(devs))

    sync_all_devices!(devs)

    return MultiRegionObject(Tuple(res), devs)
end

function sync_all_devices!(devices)
    for dev in devices
        switch_device!(dev)
        sync_device!(dev)
    end
end

sync_device!(::CuDevice) = CUDA.device_synchronize()
sync_device!(dev)        = nothing

macro apply_regionally(expr)

    apply_func! = :apply_regionally!

    if expr.head == :call
        func = expr.args[1]
        args = expr.args[2:end]
        multi_region = quote
            $apply_func!($func, $(args...))
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
                    $apply_func!($func, $(args...))
                end
            end
        end
        return quote
            $(esc(new_expr))
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
    end
end
