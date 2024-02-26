using Oceananigans.Fields: show_location, data_summary

#####
##### Show methods
#####

Base.summary(::Clamp) = "Clamp()"
Base.summary(::Linear) = "Linear()"
Base.summary(ti::Cyclical) = string("Cyclical(period=", prettysummary(ti.period), ")")

function Base.summary(fts::FieldTimeSeries{LX, LY, LZ, K}) where {LX, LY, LZ, K}
    arch = architecture(fts)
    B = string(typeof(fts.backend).name.wrapper)
    sz_str = string(join(size(fts), "×"))

    path = fts.path
    name = fts.name
    A = typeof(arch)

    if isnothing(path)
        suffix = " on $A"
    else
        suffix = " of $name at $path"
    end

    return string("$sz_str FieldTimeSeries{$B} located at ", show_location(fts), suffix)
end

function Base.show(io::IO, fts::FieldTimeSeries{LX, LY, LZ, E}) where {LX, LY, LZ, E}

    extrapolation_str = string("├── time boundaries: $(E)")

    prefix = string(summary(fts), '\n',
                   "├── grid: ", summary(fts.grid), '\n',
                   "├── indices: ", indices_summary(fts), '\n',
                   "├── time_indexing: ", summary(fts.time_indexing), '\n')

    suffix = field_time_series_suffix(fts)

    return print(io, prefix, suffix)
end

function field_time_series_suffix(fts::InMemoryFTS)
    backend = fts.backend
    backend_str = string("├── backend: ", summary(backend))
    path_str = isnothing(fts.path) ? "" : string("├── path: ", fts.path, '\n')
    name_str = isnothing(fts.name) ? "" : string("├── name: ", fts.name, '\n')

    return string(backend_str, '\n',
                  path_str,
                  name_str,
                  "└── data: ", summary(fts.data), '\n',
                  "    └── ", data_summary(parent(fts)))
end

field_time_series_suffix(fts::OnDiskFTS) =
    string("├── backend: ", summary(fts.backend), '\n',
           "├── path: ", fts.path, '\n',
           "└── name: ", fts.name)

