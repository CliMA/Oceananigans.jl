using Base: @propagate_inbounds

using OffsetArrays
using JLD2

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields

using Oceananigans.Grids: topology, total_size, interior_parent_indices, parent_index_range
using Oceananigans.Fields: show_location, interior_view_indices, data_summary

import Oceananigans.Fields: Field, set!, interior
import Oceananigans.Architectures: architecture

struct FieldTimeSeries{LX, LY, LZ, K, I, D, G, T, B, χ} <: AbstractField{LX, LY, LZ, G, T, 4}
                   data :: D
                   grid :: G
    boundary_conditions :: B
                indices :: I
                  times :: χ

    function FieldTimeSeries{LX, LY, LZ, K}(data::D, grid::G, bcs::B,
                                            times::χ, indices::I) where {LX, LY, LZ, K, D, G, B, χ, I}
        T = eltype(data) 
        return new{LX, LY, LZ, K, I, D, G, T, B, χ}(data, grid, bcs, indices, times)
    end
end

architecture(fts::FieldTimeSeries) = architecture(fts.grid)

#####
##### Constructors
#####

"""
    FieldTimeSeries{LX, LY, LZ}(grid, times, [FT=eltype(grid);]
                                indices = (:, :, :),
                                boundary_conditions = nothing)

Return a `FieldTimeSeries` at location `(LX, LY, LZ)`, on `grid`, at `times`.
"""
function FieldTimeSeries{LX, LY, LZ}(grid, times, FT=eltype(grid);
                                     indices = (:, :, :),
                                     boundary_conditions = nothing) where {LX, LY, LZ}

    Nt = length(times)
    arch = architecture(grid)
    loc = (LX, LY, LZ)
    space_size = total_size(loc, grid, indices)
    underlying_data = zeros(FT, arch, space_size..., Nt)
    data = offset_data(underlying_data, grid, loc, indices)

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

- `grid`: A grid to associated with data, in the case that the native grid was not serialized
          properly.

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
    isnothing(iterations)   && (iterations = parse.(Int, keys(file["timeseries/t"])))
    isnothing(times)        && (times      = [file["timeseries/t/$i"] for i in iterations])
    isnothing(location)     && (location   = file["timeseries/$name/serialized/location"])

    if boundary_conditions isa UnspecifiedBoundaryConditions
        boundary_conditions = file["timeseries/$name/serialized/boundary_conditions"]
    end

    indices = try
        file["timeseries/$name/serialized/indices"]
    catch
        (:, :, :)
    end

    isnothing(grid) && (grid = file["serialized/grid"])

    # Default to CPU if neither architecture nor grid is specified
    architecture = isnothing(architecture) ? (isnothing(grid) ? CPU() : Architectures.architecture(grid)) : architecture

    # This should be removed in a month or two (4/5/2022).
    grid = try
        on_architecture(architecture, grid)
    catch err # Likely, the grid has CuArrays in it...
        if grid isa RectilinearGrid # we can try...
            Nx = file["grid/Nx"]
            Ny = file["grid/Ny"]
            Nz = file["grid/Nz"]
            Hx = file["grid/Hx"]
            Hy = file["grid/Hy"]
            Hz = file["grid/Hz"]
            xᶠᵃᵃ = file["grid/xᶠᵃᵃ"]
            yᵃᶠᵃ = file["grid/yᵃᶠᵃ"]
            zᵃᵃᶠ = file["grid/zᵃᵃᶠ"]
            x = file["grid/Δxᶠᵃᵃ"] isa Number ? (xᶠᵃᵃ[1], xᶠᵃᵃ[Nx+1]) : xᶠᵃᵃ
            y = file["grid/Δyᵃᶠᵃ"] isa Number ? (yᵃᶠᵃ[1], yᵃᶠᵃ[Ny+1]) : yᵃᶠᵃ
            z = file["grid/Δzᵃᵃᶠ"] isa Number ? (zᵃᵃᶠ[1], zᵃᵃᶠ[Nz+1]) : zᵃᵃᶠ
            topo = topology(grid)

            # Reduce for Flat dimensions
            domain = NamedTuple((:x, :y, :z)[i] => (x, y, z)[i] for i=1:3 if topo[i] !== Flat)
            size = Tuple((Nx, Ny, Nz)[i] for i=1:3 if topo[i] !== Flat)
            halo = Tuple((Hx, Hy, Hz)[i] for i=1:3 if topo[i] !== Flat)

            RectilinearGrid(architecture; size, halo, topology=topo, domain...)
        else
            throw(err)
        end
    end

    close(file)

    LX, LY, LZ = location
    time_series = FieldTimeSeries{LX, LY, LZ}(grid, times; indices, boundary_conditions)

    set!(time_series, path, name)

    return time_series
end

Base.parent(fts::FieldTimeSeries) = parent(fts.data)

function Base.getindex(fts::InMemoryFieldTimeSeries, n::Int)
    underlying_data = view(parent(fts), :, :, :, n) 
    data = offset_data(underlying_data, fts.grid, location(fts), fts.indices)
    boundary_conditions = fts.boundary_conditions
    indices = fts.indices
    return Field(location(fts), fts.grid; data, boundary_conditions, indices)
end

#####
##### set!
#####

"""
    Field(location, path, name, iter;
          grid = nothing,
          architecture = nothing,
          indices = (:, :, :),
          boundary_conditions = nothing)

Load a field called `name` saved in a JLD2 file at `path` at `iter`ation.
Unless specified, the `grid` is loaded from `path`.
"""
function Field(location, path::String, name::String, iter;
               grid = nothing,
               architecture = nothing,
               indices = (:, :, :),
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

    data = offset_data(raw_data, grid, location, indices)
    
    return Field(location, grid; boundary_conditions, indices, data)
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
                        indices = time_series.indices,
                        boundary_conditions = time_series.boundary_conditions,
                        grid = time_series.grid)

        set!(time_series[n], field_n)
    end

    return nothing
end

function set!(fts::FieldTimeSeries, fields_vector::AbstractVector{<:AbstractField})
    raw_data = parent(fts)
    file = jldopen(path)

    for (n, field) in enumerate(fields_vector)
        raw_data[:, :, :, n] .= parent(field)
    end

    close(file)

    return nothing
end

function interior(fts::FieldTimeSeries)
    loc = location(fts)
    topo = topology(fts.grid)
    sz = size(fts.grid)
    halo_sz = halo_size(fts.grid)

    i_interior = interior_parent_indices.(loc, topo, sz, halo_sz)

    indices = fts.indices
    i_view = interior_view_indices.(indices, i_interior)

    return view(parent(fts), i_view..., :)
end

interior(fts::FieldTimeSeries, I...) = view(interior(fts), I...)

#####
##### OnDisk time serieses
#####

struct OnDiskData
    path :: String
    name :: String
end

function FieldTimeSeries(path, name, backend::OnDisk; architecture=nothing, grid=nothing)
    file = jldopen(path)

    if isnothing(grid)
        grid = on_architecture(architecture, file["serialized/grid"])
    end

    iterations = parse.(Int, keys(file["timeseries/t"]))
    times = [file["timeseries/t/$i"] for i in iterations]

    data = OnDiskData(path, name)
    LX, LY, LZ = file["timeseries/$name/serialized/location"]
    bcs = file["timeseries/$name/serialized/boundary_conditions"]
    indices = file["timeseries/$name/serialized/indices"]

    close(file)

    return FieldTimeSeries{LX, LY, LZ, OnDisk}(data, grid, bcs, times, indices)
end

#####
##### Methods
#####

# Include the time dimension.
@inline Base.size(fts::FieldTimeSeries) = (size(location(fts), fts.grid, fts.indices)..., length(fts.times))

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
    field_data = offset_data(raw_data, fts.grid, loc, fts.indices)

    return Field(loc, fts.grid; indices=fts.indices, boundary_conditions=fts.boundary_conditions, data=field_data)
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

function Base.summary(fts::FieldTimeSeries{LX, LY, LZ, K}) where {LX, LY, LZ, K}
    arch = architecture(fts)
    A = typeof(arch)
    return string("$(join(size(fts), "×")) FieldTimeSeries{$(backend_str(K()))} located at ", show_location(fts), " on ", A)
end

function Base.show(io::IO, fts::FieldTimeSeries)
    prefix = string(summary(fts), '\n',
                   "├── grid: ", summary(fts.grid), '\n',
                   "├── indices: ", fts.indices, '\n')

    suffix = string("└── data: ", summary(fts.data), '\n',
                    "    └── ", data_summary(fts))

    return print(io, prefix, suffix)
end
