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
