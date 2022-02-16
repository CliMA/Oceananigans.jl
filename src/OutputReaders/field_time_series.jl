using Base: @propagate_inbounds

using OffsetArrays
using JLD2

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields

using Oceananigans.Grids: topology, total_size, interior_parent_indices, parent_index_range
using Oceananigans.Fields: show_location

import Oceananigans.Fields: Field, set!, interior
import Oceananigans.Architectures: architecture

struct FieldTimeSeries{LX, LY, LZ, K, I, D, G, T, B, χ} <: AbstractField{LX, LY, LZ, G, T, 4}
                   data :: D
                   grid :: G
    boundary_conditions :: B
                indices :: I
                  times :: χ

    function FieldTimeSeries{LX, LY, LZ, K}(data::D, grid::G, bcs::B, times::χ,
                                            indices::I) where {LX, LY, LZ, K, D, G, B, χ, I}
        T = eltype(data) 
        return new{LX, LY, LZ, K, I, D, G, T, B, χ}(data, grid, bcs, times)
    end
end

architecture(fts::FieldTimeSeries) = architecture(fts.grid)

#####
##### Constructors
#####

"""
    FieldTimeSeries{LX, LY, LZ}(grid, times, boundary_conditions=nothing)

Return `FieldTimeSeries` at location `(LX, LY, LZ)`, on `grid`, at `times`, with
`boundary_conditions`, and initialized with zeros of `eltype(grid)`.
"""
function FieldTimeSeries{LX, LY, LZ}(grid, times, FT=eltype(grid);
                                     indices=(:, :, :),
                                     boundary_conditions=nothing) where {LX, LY, LZ}
    location = (LX, LY, LZ)
    Nt = length(times)
    data = offset_data(FT, grid, location, indices)
    return FieldTimeSeries{LX, LY, LZ, InMemory}(data, grid, boundary_conditions, times, indices)
end

"""
    FieldTimeSeries(path, name;
                    backend = InMemory(),
                    grid = nothing,
                    iterations = nothing,
                    times = nothing)

Returns a `FieldTimeSeries` for the field `name` describing a field's time history from a JLD2 file
located at `path`.

Keyword arguments
=================

- `backend`: `InMemory()` to load data into a 4D array or `OnDisk()` to lazily load data from disk
             when indexing into `FieldTimeSeries`.

- `grid`: A grid to associated with data, in the case that the native grid
          was not serialized properly.

- `iterations`: Iterations to load. Defaults to all iterations found in the file.

- `times`: Save times to load, as determined through an approximate floating point
           comparison to recorded save times. Defaults to times associated with `iterations`.
           Takes precedence over `iterations` if `times` is specified.
"""
FieldTimeSeries(path, name; backend=InMemory(), kwargs...) =
    FieldTimeSeries(path, name, backend; kwargs...)

#####
##### InMemory time serieses
#####

const InMemoryFieldTimeSeries{LX, LY, LZ} = FieldTimeSeries{LX, LY, LZ, InMemory}

struct UnspecifiedBoundaryConditions end

function FieldTimeSeries(path, name, backend::InMemory;
                         architecture = nothing,
                         grid = nothing,
                         location = nothing,
                         boundary_conditions = UnspecifiedBoundaryConditions(),
                         iterations = nothing,
                         times = nothing)

    file = jldopen(path)

    # Defaults
    isnothing(iterations)   && (iterations =  parse.(Int, keys(file["timeseries/t"])))
    isnothing(times)        && (times      =  [file["timeseries/t/$i"] for i in iterations])
    isnothing(location)     && (location   =  file["timeseries/$name/serialized/location"])

    # Default to CPU if neither architecture nor grid is specified
    architecture = isnothing(architecture) ?
        (isnothing(grid) ? CPU() : Architectures.architecture(grid)) :
        architecture

    if isnothing(grid)
        grid = file["serialized/grid"]
    end

    grid = on_architecture(architecture, grid)

    if boundary_conditions isa UnspecifiedBoundaryConditions
        boundary_conditions = file["timeseries/$name/serialized/boundary_conditions"]
    end

    LX, LY, LZ = location
    time_series = FieldTimeSeries{LX, LY, LZ}(grid, times; boundary_conditions)
    set!(time_series, path, name)

    return time_series
end

Base.parent(fts::FieldTimeSeries) = parent(fts.data)

Base.getindex(fts::InMemoryFieldTimeSeries{LX, LY, LZ}, n::Int) where {LX, LY, LZ} =
    Field{LX, LY, LZ}(fts.grid, boundary_conditions=fts.boundary_conditions, data=view(fts.data, :, :, :, n))

#####
##### set!
#####

"""
    Field(path::String, name::String, iter; grid=nothing)

Load a Field saved in JLD2 file at `path`, with `name` and at `iter`ation.
`grid` is loaded from `path` if not specified.
"""
function Field(location, path::String, name::String, iter;
               grid = nothing,
               architecture = nothing,
               boundary_conditions = nothing)

    file = jldopen(path)

    # Default to CPU if neither architecture nor grid is specified
    architecture = isnothing(architecture) ?
        (isnothing(grid) ? CPU() : Architectures.architecture(grid)) :
        architecture

    grid = isnothing(grid) ?
        on_architecture(architecture, file["serialized/grid"]) : grid

    raw_data = arch_array(architecture, file["timeseries/$name/$iter"])

    close(file)

    try
        data = offset_data(raw_data, grid, location)
        return Field(location, grid; boundary_conditions, data)
    catch
        field = Field(location, grid; boundary_conditions)
        interior(field) .= raw_data
        return field
    end
end

function set!(time_series::InMemoryFieldTimeSeries, path::String, name::String)

    file = jldopen(path)
    file_iterations = parse.(Int, keys(file["timeseries/t"]))
    file_times = [file["timeseries/t/$i"] for i in file_iterations]
    close(file)

    for (n, time) in enumerate(time_series.times)
        file_index = findfirst(t -> t ≈ time, file_times)
        file_iter = file_iterations[file_index]

        field_n = Field(location(time_series), path, name, file_iter,
                        boundary_conditions = time_series.boundary_conditions,
                        grid = time_series.grid)

        set!(time_series[n], field_n)
    end

    return nothing
end

function set!(time_series::FieldTimeSeries, fields_vector::AbstractVector{<:AbstractField})
    raw_data = parent(time_series.data)

    file = jldopen(path)

    for (n, field) in enumerate(fields_vector)
        raw_data[:, :, :, n] .= parent(field)
    end

    close(file)

    return nothing
end

# TODO: this is a bit of type-piracy (with respect to the Oceananigans.Fields module)...
# is there a better way? (Maybe put this into Oceananigans.Fields...)

# FieldTimeSeries[i] returns ViewField
const TimeSeriesViewField = Field{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                  S} where {S <: SubArray{<:Any, <:Any,
                                  A} where {A <: AbstractArray{<:Any, 4}}}

# Is this too surprising?
function Base.parent(vf::TimeSeriesViewField)
    space_indices = parent_index_range.(vf.data.indices[1:3])
    time_index = vf.data.indices[4]
    offset_time_series_data = parent(vf.data)
    underlying_time_series_data = parent(underlying_offset_data)
    return view(underlying_time_series_data, space_indices..., time_index)
end

"Returns a view of `f` that excludes halo points."
@inline interior(f::FieldTimeSeries{LX, LY, LZ}) where {LX, LY, LZ} =
    view(parent(f.data),
         interior_parent_indices(LX, topology(f, 1), f.grid.Nx, f.grid.Hx),
         interior_parent_indices(LY, topology(f, 2), f.grid.Ny, f.grid.Hy),
         interior_parent_indices(LZ, topology(f, 3), f.grid.Nz, f.grid.Hz),
         :)

#####
##### OnDisk time serieses
#####

struct OnDiskData
    path :: String
    name :: String
end

function FieldTimeSeries(path, name, backend::OnDisk; architecture=nothing, grid=nothing, indices=(:, :, :))
    file = jldopen(path)

    if isnothing(grid)
        grid = on_architecture(architecture, file["serialized/grid"])
    end

    iterations = parse.(Int, keys(file["timeseries/t"]))
    times = [file["timeseries/t/$i"] for i in iterations]

    data = OnDiskData(path, name)
    LX, LY, LZ = file["timeseries/$name/serialized/location"]
    bcs = file["timeseries/$name/serialized/boundary_conditions"]

    close(file)

    return FieldTimeSeries{LX, LY, LZ, OnDisk}(data, grid, bcs, times, indices)
end

# For creating an empty `FieldTimeSeries`.
function FieldTimeSeries(grid, location, times; name="", filepath="", bcs=nothing)
    LX, LY, LZ = location

    Nt = length(times)
    data_size = total_size(location, grid)

    raw_data = zeros(grid, data_size..., Nt)
    data = offset_data(raw_data, grid, location)

    return FieldTimeSeries{LX, LY, LZ}(InMemory(), data, grid, bcs, times, name, filepath, 4)
end

#####
##### Methods
#####

# Include the time dimension.
@inline Base.size(fts::FieldTimeSeries) = (size(location(fts), fts.grid)..., length(fts.times))

@propagate_inbounds Base.getindex(f::FieldTimeSeries{LX, LY, LZ, InMemory}, i, j, k, n) where {LX, LY, LZ} = f.data[i, j, k, n]

function Base.getindex(fts::FieldTimeSeries{LX, LY, LZ, OnDisk}, n::Int) where {LX, LY, LZ}
    # Load data
    arch = architecture(fts)
    file = jldopen(fts.data.path)
    iter = keys(file["timeseries/t"])[n]
    raw_data = arch_array(architecture(fts), file["timeseries/$(fts.data.name)/$iter"])
    close(file)

    # Wrap Field
    loc = (LX, LY, LZ)
    field_data = offset_data(raw_data, fts.grid, loc)

    return Field(loc, fts.grid; boundary_conditions=fts.boundary_conditions, data=field_data)
end

Base.setindex!(fts::FieldTimeSeries, val, inds...) = Base.setindex!(fts.data, val, inds...)

Base.parent(fts::FieldTimeSeries{LX, LY, LZ, OnDisk}) where {LX, LY, LZ} = nothing

#####
##### Show methods
#####

backend_str(::InMemory) = "InMemory"
backend_str(::OnDisk) = "OnDisk"

#####
##### show
#####

Base.summary(fts::FieldTimeSeries{LX, LY, LZ, K}) where {LX, LY, LZ, K} =
    string("$(join(size(fts), "×")) FieldTimeSeries{$(backend_str(K()))} located at $(show_location(fts))")

Base.show(io::IO, fts::FieldTimeSeries{LX, LY, LZ, K, A}) where {LX, LY, LZ, K, A} =
    print(io, "$(summary(fts))\n",
          "├── architecture: $A\n",
          "└── grid: $(summary(fts.grid))")

