
new_data(FT, grid, loc, indices, Nt, ::OnDisk) = nothing

Base.parent(fts::OnDiskFieldTimeSeries) = nothing
Base.length(fts::OnDiskFieldTimeSeries) = length(fts.times)

function Base.getindex(fts::FieldTimeSeries{LX, LY, LZ, OnDisk}, n::Int) where {LX, LY, LZ}
    # Load data
    arch = architecture(fts)
    file = jldopen(fts.backend.path)
    iter = keys(file["timeseries/t"])[n]
    raw_data = arch_array(arch, file["timeseries/$(fts.backend.name)/$iter"])
    close(file)

    # Wrap Field
    loc = (LX, LY, LZ)
    field_data = offset_data(raw_data, fts.grid, loc, fts.indices)

    return Field(loc, fts.grid; indices=fts.indices, boundary_conditions=fts.boundary_conditions, data=field_data)
end

#####
##### set!
#####

# When we set! a OnDiskFieldTimeSeries we automatically write down the memory path
function set!(time_series::OnDiskFieldTimeSeries, f::Field, index::Int)
    path = time_series.backend.path
    name = time_series.backend.name
    jldopen(path, "a+") do file
        initialize_file!(file, name, time_series)
        maybe_write_property!(file, "timeseries/t/$index", time_series.times[index])
        maybe_write_property!(file, "timeseries/$(name)/$(index)", parent(f))
    end
end

function initialize_file!(file, name, time_series)
    maybe_write_property!(file, "serialized/grid", time_series.grid)
    maybe_write_property!(file, "timeseries/$(name)/serialized/location", location(time_series))
    maybe_write_property!(file, "timeseries/$(name)/serialized/indices", indices(time_series))
    maybe_write_property!(file, "timeseries/$(name)/serialized/boundary_conditions", boundary_conditions(time_series))
    return nothing
end

# Write property only if it does not already exist
function maybe_write_property!(file, property, data)
    try
        test = file[property]
    catch 
        file[property] = data
    end
end

set!(time_series::OnDiskFieldTimeSeries, path::String, name::String) = nothing