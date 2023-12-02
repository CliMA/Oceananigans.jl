
new_data(FT, grid, loc, indices, Nt, ::OnDisk) = nothing

Base.parent(fts::OnDiskFieldTimeSeries) = nothing
Base.length(fts::OnDiskFieldTimeSeries) = length(fts.times)

function Base.getindex(fts::FieldTimeSeries{LX, LY, LZ, OnDisk}, n::Int) where {LX, LY, LZ}
    # Load data
    arch = architecture(fts)
    file = jldopen(fts.path)
    iter = keys(file["timeseries/t"])[n]
    raw_data = arch_array(arch, file["timeseries/$(fts.name)/$iter"])
    close(file)

    # Wrap Field
    loc = (LX, LY, LZ)
    field_data = offset_data(raw_data, fts.grid, loc, fts.indices)

    return Field(loc, fts.grid;
                 indices = fts.indices,
                 boundary_conditions = fts.boundary_conditions,
                 data = field_data)
end

#####
##### set!
#####

# Write property only if it does not already exist
function maybe_write_property!(file, property, data)
    try
        test = file[property]
    catch 
        file[property] = data
    end
end

"""
    set!(fts::OnDiskFieldTimeSeries, field::Field, time_index::Int)

Write the data in `parent(field)` to the file at `fts.path`,
under `fts.name` and at index `fts.times[time_index]`.
"""
function set!(fts::OnDiskFieldTimeSeries, field::Field, time_index::Int)
    path = fts.path
    name = fts.name
    jldopen(path, "a+") do file
        initialize_file!(file, name, fts)
        maybe_write_property!(file, "timeseries/t/$time_index", fts.times[time_index])
        maybe_write_property!(file, "timeseries/$name/$time_index", parent(field))
    end
end

function initialize_file!(file, name, fts)
    maybe_write_property!(file, "serialized/grid", fts.grid)
    maybe_write_property!(file, "timeseries/$name/serialized/location", location(fts))
    maybe_write_property!(file, "timeseries/$name/serialized/indices", indices(fts))
    maybe_write_property!(file, "timeseries/$name/serialized/boundary_conditions", boundary_conditions(fts))
    return nothing
end

set!(fts::OnDiskFieldTimeSeries, path::String, name::String) = nothing
fill_halo_regions!(fts::OnDiskFieldTimeSeries) = nothing

