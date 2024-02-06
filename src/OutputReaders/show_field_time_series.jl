#####
##### Show methods
#####

backend_str(::Type{InMemory}) = "InMemory"
backend_str(::Type{OnDisk})   = "OnDisk"

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
                   "├── time extrapolation: $(fts.time_extrapolation)", '\n')

    suffix = field_time_series_suffix(fts)

    return print(io, prefix, suffix)
end

function field_time_series_suffix(fts::InMemoryFieldTimeSeries)
    ii = fts.backend.indices

    if ii isa Colon
        backend_str = "├── backend: InMemory(:)"
    else
        N = length(ii)
        if N < 6
            indices_str = string(ii)
        else
            indices_str = string("[", ii[1],
                                 ", ", ii[2],
                                 ", ", ii[3],
                                 "  …  ",
                                 ii[end-2], ", ",
                                 ii[end-1], ", ",
                                 ii[end], "]")
        end

        backend_str = string("├── backend: InMemory(", indices_str, ")")
    end

    path_str = isnothing(fts.path) ? "" : string("├── path: ", fts.path, '\n')
    name_str = isnothing(fts.name) ? "" : string("├── name: ", fts.name, '\n')

    return string(backend_str, '\n',
                  path_str,
                  name_str,
                  "└── data: ", summary(fts.data), '\n',
                  "    └── ", data_summary(interior(fts)))
end

field_time_series_suffix(fts::OnDiskFieldTimeSeries) =
    string("├── backend: ", summary(fts.backend), '\n',
           "├── path: ", fts.path, '\n',
           "└── name: ", fts.name)

