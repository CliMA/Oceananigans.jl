struct FieldDataset{F, M, P}
    fields :: F
  metadata :: M
  filepath :: P
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
"""
function FieldDataset(filepath;
                    architecture=CPU(), grid=nothing, backend=InMemory(), metadata_paths=["metadata"])

  file = jldopen(filepath)

  field_names = keys(file["timeseries"])
  filter!(k -> k != "t", field_names)  # Time is not a field.

  ds = Dict{String, FieldTimeSeries}(
      name => FieldTimeSeries(filepath, name; architecture, backend, grid)
      for name in field_names
  )

  metadata = Dict(
      k => file["$mp/$k"]
      for mp in metadata_paths if haskey(file, mp)
      for k in keys(file["$mp"])
  )

  close(file)

  return FieldDataset(ds, metadata, abspath(filepath))
end

Base.getindex(fds::FieldDataset, inds...) = Base.getindex(fds.fields, inds...)

Base.show(io::IO, fds::FieldDataset) =
  print(io, "FieldDataset with $(length(fds.fields)) fields and $(length(fds.metadata)) metadata entries.")
