
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

Base.getindex(mo::MultiRegionObject, args...) = Base.getindex(mo.regions, args...)

switch_device!(mo::MultiRegionObject, i) = switch_device!(getdevice(mo, i))

getregion(mo::MultiRegionObject, i) = mo.regions[i]
getdevice(mo::MultiRegionObject, i) = mo.devices[i]

regions(mo::MultiRegionObject) = 1:length(mo.regions)
devices(mo::MultiRegionObject) = mo.devices

isregional(mo::MultiRegionObject) = true

iterate_args(::Nothing, args...)               = (nothing, nothing)
iterate_args(args::Tuple, iterate, idx)        = Tuple(to_iterate(args[i], iterate[i], idx) for i in 1:length(args))
iterate_args(kwargs::NamedTuple, iterate, idx) = NamedTuple{keys(kwargs)}(to_iterate(kwargs[i], iterate[i], idx) for i in 1:length(kwargs))

to_iterate(arg, iter, idx) = iter == 1 ? arg[idx] : arg
