using Base: @propagate_inbounds

using OffsetArrays
using JLD2

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields

using Oceananigans.Grids: topology, total_size, interior_parent_indices
using Oceananigans.Fields: show_location

import Oceananigans: short_show
import Oceananigans.Fields: Field, set!, interior

struct FieldTimeSeries{X, Y, Z, K, A, T, D, G, B, χ} <: AbstractDataField{X, Y, Z, A, G, T, 4}
                   data :: D
           architecture :: A
                   grid :: G
    boundary_conditions :: B
                  times :: χ

    function FieldTimeSeries{X, Y, Z, K}(data::D, arch::A, grid::G, bcs::B, times::χ) where {X, Y, Z, K, D, A, G, B, χ}
        T = eltype(grid) 
        return new{X, Y, Z, K, A, T, D, G, B, χ}(data, arch, grid, bcs, times)
    end
end

#####
##### Constructors
#####

"""
    FieldTimeSeries{LX, LY, LZ}([architecture = CPU()], grid, times, boundary_conditions=nothing)

Return `FieldTimeSeries` at location `(LX, LY, LZ)`, on `grid`, at `times`, with
`boundary_conditions`, and initialized with zeros of `eltype(grid)`.
"""
function FieldTimeSeries{LX, LY, LZ}(architecture, grid, times, boundary_conditions=nothing) where {LX, LY, LZ}
    location = (LX, LY, LZ)
    Nt = length(times)
    data_size = total_size(location, grid)
    raw_data = zeros(architecture, grid, data_size..., Nt)
    data = offset_data(raw_data, grid, location)
    return FieldTimeSeries{LX, LY, LZ, InMemory}(data, architecture, grid, boundary_conditions, times)
end

# CPU() default
FieldTimeSeries{LX, LY, LZ}(grid::AbstractGrid, times, bcs=nothing) where {LX, LY, LZ} =
    FieldTimeSeries{LX, LY, LZ}(CPU(), grid, times, bcs)

"""
    FieldTimeSeries(path, name;
                    architecture = CPU(),
                    backend = InMemory(),
                    grid = nothing,
                    iterations = nothing,
                    times = nothing)

Returns a `FieldTimeSeries` for the field `name` describing a field's time history from a JLD2 file
located at `path`.

Keyword arguments
=================

- `architecture`: The architecture on which to store time series data. `CPU()` by default.

- `backend`: `InMemory()` to load data into a 4D array or `OnDisk()` to lazily load data from disk
             when indexing into `FieldTimeSeries`.

- `grid`: A grid to associated with data, in the case that the native grid
          was not serialized properly.

- `iterations`: Iterations to load. Defaults to all iterations found in the file.

- `times`: Save times to load, as determined through an approximate floating point
           comparison to recorded save times. Defaults to times associated with `iterations`.
           Takes precedence over `iterations` if `times` is specified.
"""
FieldTimeSeries(path, name; architecture=CPU(), backend=InMemory(), kwargs...) =
    FieldTimeSeries(path, name, architecture, backend; kwargs...)

#####
##### InMemory time serieses
#####

const InMemoryFieldTimeSeries{X, Y, Z} = FieldTimeSeries{X, Y, Z, InMemory}

struct UnspecifiedBoundaryConditions end

function FieldTimeSeries(path, name, architecture, backend::InMemory;
                         grid = nothing,
                         location = nothing,
                         boundary_conditions = UnspecifiedBoundaryConditions(),
                         iterations = nothing,
                         times = nothing)

    file = jldopen(path)

    # Non-defaults
    isnothing(grid)       && (grid       =  file["serialized/grid"])
    isnothing(iterations) && (iterations =  parse.(Int, keys(file["timeseries/t"])))
    isnothing(times)      && (times      =  [file["timeseries/t/$i"] for i in iterations])
    isnothing(location)   && (location   =  file["timeseries/$name/serialized/location"])

    if boundary_conditions isa UnspecifiedBoundaryConditions
        boundary_conditions = file["timeseries/$name/serialized/boundary_conditions"]
    end

    LX, LY, LZ = location
  
    time_series = FieldTimeSeries{LX, LY, LZ}(architecture, grid, times, boundary_conditions)

    set!(time_series, path, name)
    
    return time_series
end

Base.getindex(fts::InMemoryFieldTimeSeries{LX, LY, LZ}, n::Int) where {LX, LY, LZ} =
    Field(LX, LY, LZ, fts.architecture, fts.grid, fts.boundary_conditions, view(fts.data, :, :, :, n))

#####
##### set!
#####

"""
    Field(path::String, name::String, iter; architecture=GPU(), grid=nothing)

Load a Field saved in JLD2 file at `path`, with `name` and at `iter`ation.
`architecture = CPU()` by default, and `grid` is loaded from `path` if not specified.
"""
function Field(location, path::String, name::String, iter;
               architecture = CPU(),
               grid = nothing,
               boundary_conditions = nothing)

    file = jldopen(path)

    raw_data = arch_array(architecture, file["timeseries/$name/$iter"])
    isnothing(grid) && (grid = file["serialized/grid"])

    close(file)

    data = offset_data(raw_data, grid, location)

    return Field(location, architecture, grid, boundary_conditions, data)
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
    ArrayType = array_type(time_series.architecture)

    file = jldopen(path)

    for (n, field) in enumerate(fields_vector)
        raw_data[:, :, :, n] .= parent(field)
    end

    close(file)

    return nothing
end

# TODO: this is a bit of type-piracy (with respect to the Oceananigans.Fields module)...
# is there a better way?

# FieldTimeSeries[i] returns ViewField
const ViewField = Field{<:Any, <:Any, <:Any, <:Any, <:SubArray}

using OffsetArrays: IdOffsetRange

parent_indices(idx::Int) = idx
parent_indices(idx::Base.Slice{<:IdOffsetRange}) = Colon()

# Is this too surprising?
Base.parent(vf::ViewField) = view(parent(parent(vf.data)), parent_indices.(vf.data.indices)...)

"Returns a view of `f` that excludes halo points."
@inline interior(f::FieldTimeSeries{X, Y, Z}) where {X, Y, Z} =
    view(parent(f.data),
         interior_parent_indices(X, topology(f, 1), f.grid.Nx, f.grid.Hx),
         interior_parent_indices(Y, topology(f, 2), f.grid.Ny, f.grid.Hy),
         interior_parent_indices(Z, topology(f, 3), f.grid.Nz, f.grid.Hz),
         :)

#####
##### OnDisk time serieses
#####

struct OnDiskData
    path :: String
    name :: String
end

function FieldTimeSeries(path, name, architecture, backend::OnDisk; grid=nothing)
    file = jldopen(path)

    if isnothing(grid)
        grid = file["serialized/grid"]
    end

    iterations = parse.(Int, keys(file["timeseries/t"]))
    times = [file["timeseries/t/$i"] for i in iterations]

    data = OnDiskData(path, name)
    LX, LY, LZ = file["timeseries/$name/serialized/location"]
    bcs = file["timeseries/$name/serialized/boundary_conditions"]

    close(file)

    return FieldTimeSeries{LX, LY, LZ, OnDisk}(data, architecture, grid, bcs, times)
end

# For creating an empty `FieldTimeSeries`.
function FieldTimeSeries(grid, location, times; architecture=CPU(), ArrayType=array_type(architecture), name="", filepath="", bcs=nothing)
    LX, LY, LZ = location

    Nt = length(times)
    data_size = total_size(location, grid)

    raw_data = zeros(data_size..., Nt) |> ArrayType
    data = offset_data(raw_data, grid, location)

    return FieldTimeSeries{LX, LY, LZ}(InMemory(), data, architecture, grid, bcs, times, name, filepath, 4)
end

#####
##### Methods
#####

# Include the time dimension.
@inline Base.size(fts::FieldTimeSeries) = (size(location(fts), fts.grid)..., length(fts.times))

@propagate_inbounds Base.getindex(f::FieldTimeSeries{X, Y, Z, InMemory}, i, j, k, n) where {X, Y, Z} = f.data[i, j, k, n]

function Base.getindex(fts::FieldTimeSeries{X, Y, Z, OnDisk}, n::Int) where {X, Y, Z}
    # Load data
    file = jldopen(fts.data.path)
    iter = keys(file["timeseries/t"])[n]
    raw_data = file["timeseries/$(fts.data.name)/$iter"] |> array_type(fts.architecture)
    close(file)

    # Wrap Field
    loc = (X, Y, Z)
    field_data = offset_data(raw_data, fts.grid, loc)

    return Field(loc..., fts.architecture, fts.grid, fts.boundary_conditions, field_data)
end

Base.setindex!(fts::FieldTimeSeries, val, inds...) = Base.setindex!(fts.data, val, inds...)

#####
##### Show methods
#####

backend_str(::InMemory) = "InMemory"
backend_str(::OnDisk) = "OnDisk"

#####
##### show
#####

short_show(fts::FieldTimeSeries{X, Y, Z, K}) where {X, Y, Z, K} =
    string("$(join(size(fts), "×")) FieldTimeSeries{$(backend_str(K()))} located at $(show_location(fts))")

Base.show(io::IO, fts::FieldTimeSeries{X, Y, Z, K, A}) where {X, Y, Z, K, A} =
    print(io, "$(short_show(fts))\n",
          "├── architecture: $A\n",
          "└── grid: $(short_show(fts.grid))")

