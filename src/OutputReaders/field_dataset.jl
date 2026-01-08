using Oceananigans.Fields: instantiated_location, indices
import Oceananigans.Fields: set!

struct FieldDataset{F, B, M, P, KW}
        fields :: F
       backend :: B
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

  return FieldDataset(ds, backend, metadata, abspath(filepath), reader_kw)
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

"""
    FieldDataset(grid, times, fields;
                 backend = OnDisk(),
                 path = nothing,
                 location = NamedTuple(),
                 indices = NamedTuple(),
                 boundary_conditions = NamedTuple(),
                 metadata = Dict(),
                 reader_kw = NamedTuple())

Returns a `FieldDataset` containing a new `FieldTimeSeries` for each key in `fields`
on `grid` at `times`.

Keyword arguments
=================
- `backend`: backend, `InMemory(indices=Colon())` or `OnDisk()`

- `path`: path to data for `backend = OnDisk()`

- `location`: `NamedTuple` of location specifications, defaults to 
               (Center, Center, Center) for each field`

- `indices`: `NamedTuple` of spatial indices, defaults to (:, :, :) for each field

- `boundary_conditions`: `NamedTuple` of boundary conditions for each field

- `metadata`: `Dict` containing metadata entries
"""
function FieldDataset(grid, times, fields::NTuple{N, Symbol};
    backend=OnDisk(),
    path=nothing,
    location=NamedTuple(),
    indices=NamedTuple(),
    boundary_conditions=NamedTuple(),
    metadata=Dict(),
    reader_kw=NamedTuple(),
    ) where {N}

    field_names = map(String, fields)

    # Default behaviour
    indices = merge(
        NamedTuple(field=>(:, :, :) for field in fields),
        indices
    )
    location = merge(
        NamedTuple(field=>(Center(), Center(), Center()) for field in fields),
        location
    )
    boundary_conditions = merge(
        NamedTuple(field=>UnspecifiedBoundaryConditions() for field in fields),
        boundary_conditions
    )

    # Create the FieldTimeSeries
    ftss = map(fields, field_names) do field, name
        inds = indices[field]
        loc = location[field]
        bcs = boundary_conditions[field]

        FieldTimeSeries(loc, grid, times; 
            indices=inds, 
            backend, 
            path, 
            name, 
            reader_kw, 
            boundary_conditions=bcs
        )
    end

    ds = Dict{String, FieldTimeSeries}(
        name => fts
        for (name, fts) in zip(field_names, ftss)
    )

    return FieldDataset(ds, backend, metadata, path, reader_kw)
end


"""
    FieldDataset(times, fields;
                 backend = OnDisk(),
                 path = nothing,
                 metadata = Dict(),
                 reader_kw = NamedTuple())

Returns a `FieldDataset` containing a new `FieldTimeSeries` for each field 
in the `NamedTuple``fields` at `times`. Locations, indices and boundary 
conditions are extracted from `fields``

Keyword arguments
=================
- `backend`: backend, `InMemory(indices=Colon())` or `OnDisk()`

- `path`: path to data for `backend = OnDisk()`

- `metadata`: `Dict` containing metadata entries
"""
function FieldDataset(times, fields;
    fds_kw...
    )

    grid = fields[1].grid
    any([field.grid != grid for field in fields]) && throw(ArgumentError("All fields must be defined on the same grid"))

    loc = map(instantiated_location, fields)
    inds = map(Fields.indices, fields)
    bcs = map(boundary_conditions, fields)

    return FieldDataset(grid, times, keys(fields);
        location=loc,
        indices=inds,
        boundary_conditions=bcs,
        fds_kw...
    )
end

# Setting a FieldDataset iterates over contained FieldTimeSeries
function set!(fds::FieldDataset, args...; fields...)
    for (k, v) in pairs(fields)
        set!(fds[k], v, args...)
    end
end

# Write metadata if possible for OnDisk FieldDataset
function set!(fds::FieldDataset{F, B, M, P, KW}, args...; fields...) where {F, B<:OnDisk, M, P, KW}
    jldopen(fds.filepath, "a+") do file
        for (k, v) in pairs(fds.metadata)
            maybe_write_property!(file, "metadata/$k", v)
        end
    end
    for (k, v) in pairs(fields)
        set!(fds[k], v, args...)
    end
end
