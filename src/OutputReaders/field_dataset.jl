"""
    FieldTimeSeries(filepath, name; architecture=CPU(), backend=InMemory())

Returns a `Dict` containing a `FieldTimeSeries` for each field in the JLD2 file located at `filepath`.
Note that model output must have been saved with halos. The `InMemory` backend will store the data
fully in memory as a 4D multi-dimensional array while the `OnDisk` backend will lazily load field time
snapshots when the `FieldTimeSeries` is indexed linearly.
"""
function FieldDataset(filepath; architecture=CPU(), backend=InMemory())
    file = jldopen(filepath)

    field_names = keys(file["timeseries"])
    filter!(k -> k != "t", field_names)  # Time is not a field.

    ds = Dict{String, FieldTimeSeries}(
        name => FieldTimeSeries(filepath, name; architecture, backend)
        for name in field_names
    )

    close(file)

    return ds
end
