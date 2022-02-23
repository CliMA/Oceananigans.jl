struct MultiRegionObject{R, D}
    regions :: R
    devices :: D
end

MultiRegionObject(::Nothing, devices) = nothing
MultiRegionObject(::NTuple{N, Nothing}, devices) where N = nothing

Base.getindex(mo::MultiRegionObject, args...) = Base.getindex(mo.regions, args...)

switch_device!(mo::MultiRegionObject, i) = switch_device!(getdevice(mo, i))

getregion(mo::MultiRegionObject, i) = mo.regions[i]
getdevice(mo::MultiRegionObject, i) = mo.devices[i]

Adapt.adapt_structure(to, mo::MultiRegionObject) = 
                MultiRegionObject(Adapt.adapt(to, regions),  nothing)

regions(mo::MultiRegionObject) = 1:length(mo.regions)
devices(mo::MultiRegionObject) = mo.devices

isregional(mo::MultiRegionObject) = true
