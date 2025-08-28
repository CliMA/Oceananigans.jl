struct FieldDataset{F, M, P, KW}
        fields :: F
      metadata :: M
      filepath :: P
    reader_kw :: KW
end

"""
    FieldDataset(filepath;
                 architecture=CPU(), grid=nothing, backend=InMemory(), metadata_paths=["metadata"])

Returns a `Dict` containing a `FieldTimeSeries` for each field in the JLD2 file located
at `filepath`. Note that model output **must** have been saved with halos.

Keyword arguments
=================
- `backend`: Either `InMemory()` (default) or `OnDisk()`. The `InMemory` backend will
load the data fully in memory as a 4D multi-dimensional array while the `OnDisk()`
backend will lazily load field time snapshots when the `FieldTimeSeries` is indexed
linearly.

- `metadata_paths`: A list of JLD2 paths to look for metadata. By default it looks in
  `file["metadata"]`.

- `grid`: May be specified to override the grid used in the JLD2 file.

- `reader_kw`: A named tuple or dictionary of keyword arguments to pass to the reader
               (currently only JLD2) to be used when opening files.
"""
function FieldDataset(filepath;
                      architecture = CPU(),
                      grid = nothing,
                      backend = InMemory(),
                      metadata_paths = ["metadata"],
                      reader_kw = NamedTuple())

  file = jldopen(filepath; reader_kw...)

  field_names = keys(file["timeseries"])
  filter!(k -> k != "t", field_names)  # Time is not a field.

  ds = Dict{String, FieldTimeSeries}(
      name => FieldTimeSeries(filepath, name; architecture, backend, grid, reader_kw)
      for name in field_names
  )

  metadata = Dict(
      k => file["$mp/$k"]
      for mp in metadata_paths if haskey(file, mp)
      for k in keys(file["$mp"])
  )

  close(file)

  return FieldDataset(ds, metadata, abspath(filepath), reader_kw)
end

Base.getindex(fds::FieldDataset, inds...) = Base.getindex(fds.fields, inds...)
Base.getindex(fds::FieldDataset, i::Symbol) = Base.getindex(fds, string(i))

Base.keys(fds::FieldDataset) = Base.keys(fds.fields)

function Base.getproperty(fds::FieldDataset, name::Symbol)
    if name in propertynames(fds)
        return getfield(fds, name)
    else
        return getindex(fds, name)
    end
end

function Base.show(io::IO, fds::FieldDataset)
    s = "FieldDataset with $(length(fds.fields)) fields and $(length(fds.metadata)) metadata entries:\n"

    n_fields = length(fds.fields)

    for (i, (name, fts)) in enumerate(pairs(fds.fields))
        prefix = i == n_fields ? "└── " : "├── "
        s *= prefix * "$name: " * summary(fts) * '\n'
    end

    return print(io, s)
end
