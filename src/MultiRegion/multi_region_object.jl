
struct MultiRegionObject{R, D}
    regions :: R
    devices :: D
end

MultiRegionObject(::Nothing, devices) = nothing
MultiRegionObject(::NTuple{N, Nothing}, devices) where N = nothing

function MultiRegionObject(devices::Tuple, constructor, args, iter_args, kwargs, iter_kwargs)
    regional_obj = []
    for (i, dev) in enumerate(devices)
        switch_device!(dev)
        if isnothing(args) & isnothing(kwargs)
            push!(regional_obj, constructor())
        elseif isnothing(kwargs)
            push!(regional_obj, constructor(iterate_args(args, iter_args, i)...))
        elseif isnothing(kwargs)
            push!(regional_obj, constructor(; iterate_args(kwargs, iter_args, i)...))
        else
            push!(regional_obj, constructor(iterate_args(args, iter_args, i)...; iterate_args(kwargs, iter_kwargs, i)...))
        end
    end
    return MultiRegionObject(Tuple(regional_obj), devices)
end

# For non-returning functions
function apply_regionally!(func!, args...; kwargs...)
    mra = isnothing(findfirst(isregional, args)) ? nothing : args[findfirst(isregional, args)]
    mrk = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
    isnothing(mra) && isnothing(mrk) && return func(args...; kwargs...)

    for (r, dev) in enumerate(devices(mra))
        switch_device!(dev);
        region_args = Tuple(getregion(arg, r) for arg in args);
        region_kwargs = NamedTuple{keys(kwargs)}(getregion(kwarg, r) for kwarg in kwargs);
        func!(region_args...; region_kwargs...)
    end
end
 
# For functions with return statements
function apply_regionally(func, args...; kwargs...)
    mra = isnothing(findfirst(isregional, args)) ? nothing : args[findfirst(isregional, args)]
    mrk = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
    isnothing(mra) && isnothing(mrk) && return func(args...; kwargs...)

    res = Tuple((switch_device!(dev);
                 region_args = Tuple(getregion(arg, r) for arg in args);
                 region_kwargs = NamedTuple{keys(kwargs)}(getregion(kwarg, r) for kwarg in kwargs);
                 func(region_args...; region_kwargs...))
                 for (r, dev) in enumerate(devices(mra)))
    
    return MultiRegionObject(res, devices(mra))
end

Base.getindex(mo::MultiRegionObject, args...) = Base.getindex(mo.regions, args...)

switch_device!(mo::MultiRegionObject, i) = switch_device!(getdevice(mo, i))

getregion(mo::MultiRegionObject, i) = mo.regions[i]
getdevice(mo::MultiRegionObject, i) = mo.devices[i]

regions(mo::MultiRegionObject) = length(mo.regions)
devices(mo::MultiRegionObject) = mo.devices

isregional(mo::MultiRegionObject) = true

iterate_args(::Nothing, args...)               = (nothing, nothing)
iterate_args(args::Tuple, iterate, idx)        = Tuple(to_iterate(args[i], iterate[i], idx) for i in 1:length(args))
iterate_args(kwargs::NamedTuple, iterate, idx) = NamedTuple{keys(kwargs)}(to_iterate(kwargs[i], iterate[i], idx) for i in 1:length(kwargs))

to_iterate(arg, iter, idx) = iter == 1 ? arg[idx] : arg