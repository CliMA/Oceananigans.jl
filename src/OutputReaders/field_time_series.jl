using Base: @propagate_inbounds

using OffsetArrays
using Statistics
using JLD2

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields

using Oceananigans.Grids: topology, total_size, interior_parent_indices, parent_index_range
using Oceananigans.Fields: show_location, interior_view_indices, data_summary, reduced_location

import Oceananigans.Fields: Field, set!, interior, indices
import Oceananigans.Architectures: architecture

struct FieldTimeSeries{LX, LY, LZ, K, I, D, G, T, B, χ} <: AbstractField{LX, LY, LZ, G, T, 4}
                   data :: D
                   grid :: G
    boundary_conditions :: B
                indices :: I
                  times :: χ

    function FieldTimeSeries{LX, LY, LZ, K}(data::D,
                                            grid::G,
                                            bcs::B,
                                            times::χ,
                                            indices::I) where {LX, LY, LZ, K, D, G, B, χ, I}
        T = eltype(data)
        return new{LX, LY, LZ, K, I, D, G, T, B, χ}(data, grid, bcs, indices, times)
    end
end

architecture(fts::FieldTimeSeries) = architecture(fts.grid)

const InMemoryFieldTimeSeries{LX, LY, LZ} = FieldTimeSeries{LX, LY, LZ, InMemory}
const OnDiskFieldTimeSeries{LX, LY, LZ} = FieldTimeSeries{LX, LY, LZ, OnDisk}

struct UnspecifiedBoundaryConditions end

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

Return a `FieldTimeSeries` containing a time-series of the field `name`
load from JLD2 output located at `path`.

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
FieldTimeSeries(path, name; backend=InMemory(), kw...) = FieldTimeSeries(path, name, backend; kw...)

instantiate(T::Type) = T()

function FieldTimeSeries(path, name, backend;
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
    isnothing(location)     && (Location   = file["timeseries/$name/serialized/location"])

    if boundary_conditions isa UnspecifiedBoundaryConditions
        boundary_conditions = file["timeseries/$name/serialized/boundary_conditions"]
    end

    indices = try
        file["timeseries/$name/serialized/indices"]
    catch
        (:, :, :)
    end

    isnothing(grid) && (grid = file["serialized/grid"])
    close(file)

    # Default to CPU if neither architecture nor grid is specified
    architecture = isnothing(architecture) ?
        (isnothing(grid) ? CPU() : Architectures.architecture(grid)) : architecture

    # This should be removed in a month or two (4/5/2022).
    grid = on_architecture(architecture, grid)

    LX, LY, LZ = Location
    loc = map(instantiate, Location)

    if backend isa InMemory
        Nt = length(times)
        space_size = total_size(grid, loc, indices)
        underlying_data = zeros(eltype(grid), architecture, space_size..., Nt)
        data = offset_data(underlying_data, grid, loc, indices)
    elseif backend isa OnDisk
        data = OnDiskData(path, name)
    else
        error("FieldTimeSeries does not support backend $backend!")
    end

    K = typeof(backend)
    time_series = FieldTimeSeries{LX, LY, LZ, K}(data, grid, boundary_conditions, times, indices)
    set!(time_series, path, name)

    return time_series
end

Base.parent(fts::InMemoryFieldTimeSeries) = parent(fts.data)
Base.parent(fts::OnDiskFieldTimeSeries) = nothing

@propagate_inbounds Base.getindex(f::FieldTimeSeries{LX, LY, LZ, InMemory}, i, j, k, n) where {LX, LY, LZ} = f.data[i, j, k, n]

function Base.getindex(fts::InMemoryFieldTimeSeries, n::Int)
    underlying_data = view(parent(fts), :, :, :, n) 
    data = offset_data(underlying_data, fts.grid, location(fts), fts.indices)
    boundary_conditions = fts.boundary_conditions
    indices = fts.indices
    return Field(location(fts), fts.grid; data, boundary_conditions, indices)
end

# Making FieldTimeSeries behave like Vector
Base.lastindex(fts::FieldTimeSeries) = size(fts, 4)
Base.firstindex(fts::FieldTimeSeries) = 1

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

#####
##### set!
#####

set!(time_series::OnDiskFieldTimeSeries, path::String, name::String) = nothing

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
    loc = instantiate.(location(fts))
    topo = instantiate.(topology(fts.grid))
    sz = size(fts.grid)
    halo_sz = halo_size(fts.grid)

    i_interior = interior_parent_indices.(loc, topo, sz, halo_sz)

    indices = fts.indices
    i_view = interior_view_indices.(indices, i_interior)

    return view(parent(fts), i_view..., :)
end

interior(fts::FieldTimeSeries, I...) = view(interior(fts), I...)
indices(fts::FieldTimeSeries) = fts.indices

function Statistics.mean(fts::FieldTimeSeries; dims=:)
    m = mean(fts[1]; dims)
    Nt = length(fts.times)

    if dims isa Colon
        for n = 2:Nt
            m += mean(fts[n])
        end

        return m / Nt
    else
        for n = 2:Nt
            m .+= mean(fts[n]; dims)
        end

        m ./= Nt

        return m
    end
end

#####
##### Methods
#####

# Include the time dimension.
@inline Base.size(fts::FieldTimeSeries) = (size(fts.grid, location(fts), fts.indices)..., length(fts.times))

Base.setindex!(fts::FieldTimeSeries, val, inds...) = Base.setindex!(fts.data, val, inds...)

#####
##### Basic support for reductions
#####
##### TODO: support for reductions across _time_ (ie when 4 ∈ dims)
#####

const FTS = FieldTimeSeries

for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)
    reduction! = Symbol(reduction, '!')

    @eval begin

        # Allocating
        function Base.$(reduction)(f::Function, fts::FTS; dims=:, kw...)
            if dims isa Colon        
                return Base.$(reduction)($(reduction)(f, fts[n]; kw...) for n in 1:length(fts.times))
            else
                T = filltype(Base.$(reduction!), fts)
                loc = LX, LY, LZ = reduced_location(location(fts); dims)
                times = fts.times
                rts = FieldTimeSeries{LX, LY, LZ}(grid, times, T; indices=fts.indices)
                return Base.$(reduction!)(f, rts, fts; kw...)
            end
        end

        Base.$(reduction)(fts::FTS; kw...) = Base.$(reduction)(identity, fts; kw...)

        function Base.$(reduction!)(f::Function,rts::FTS, fts::FTS; dims=:, kw...)
            dims isa Tuple && 4 ∈ dims && error("Reduction across the time dimension (dim=4) is not yet supported!")
            times = rts.times
            for n = 1:length(times)
                Base.$(reduction!)(f, rts[i], fts[i]; dims, kw...)
            end
            return rts
        end

        Base.$(reduction!)(rts::FTS, fts::FTS; kw...) = Base.$(reduction!)(identity, rts, fts; kw...)
    end
end

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
    prefix = string(summary(fts), "\n",
                   "├── grid: ", summary(fts.grid), "\n",
                   "├── indices: ", fts.indices, "\n")

    suffix = field_time_series_suffix(fts)

    return print(io, prefix, suffix)
end

field_time_series_suffix(fts::InMemoryFieldTimeSeries) =
    string("└── data: ", summary(fts.data), "\n",
           "    └── ", data_summary(fts))

field_time_series_suffix(fts::OnDiskFieldTimeSeries) =
    string("└── data: ", summary(fts.data))

