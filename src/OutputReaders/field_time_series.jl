using Base: @propagate_inbounds

using OffsetArrays
using Statistics
using JLD2

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields

using Oceananigans.Grids: topology, total_size, interior_parent_indices, parent_index_range
using Oceananigans.Fields: show_location, interior_view_indices, data_summary, reduced_location, index_binary_search

using Oceananigans.Fields: boundary_conditions 

using Oceananigans.OutputWriters: serializeproperty!, iteration_exists
import Oceananigans.Fields: Field, set!, interior, indices
import Oceananigans.Architectures: architecture

using Dates: AbstractTime

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

const InMemoryFieldTimeSeries{LX, LY, LZ} = FieldTimeSeries{LX, LY, LZ, <:InMemory}
const OnDiskFieldTimeSeries{LX, LY, LZ} = FieldTimeSeries{LX, LY, LZ, <:OnDisk}
const ChunkedFieldTimeSeries{LX, LY, LZ} = FieldTimeSeries{LX, LY, LZ, <:Chunked}

struct UnspecifiedBoundaryConditions end

#####
##### Constructors
#####

instantiate(T::Type) = T()

"""
    FieldTimeSeries{LX, LY, LZ}(grid, times, [FT=eltype(grid);]
                                indices = (:, :, :),
                                boundary_conditions = nothing)

Return a `FieldTimeSeries` at location `(LX, LY, LZ)`, on `grid`, at `times`.
"""
function FieldTimeSeries{LX, LY, LZ}(grid, times, FT=eltype(grid);
                                     indices = (:, :, :), 
                                     backend = InMemory(),
                                     path = nothing,
                                     name = nothing,
                                     boundary_conditions = nothing) where {LX, LY, LZ}

    Nt   = length(times)
    loc  = map(instantiate, (LX, LY, LZ))
    data = new_data(FT, grid, loc, indices, Nt, path, name, backend)
    K = typeof(backend)
    return FieldTimeSeries{LX, LY, LZ, K}(data, grid, boundary_conditions, times, indices)
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
    Nt = length(times)
    data = new_data(eltype(grid), grid, loc, indices, Nt, path, name, backend)

    K = typeof(backend)
    time_series = FieldTimeSeries{LX, LY, LZ, K}(data, grid, boundary_conditions, times, indices)
    set!(time_series, path, name)

    return time_series
end

# Making FieldTimeSeries behave like Vector
Base.lastindex(fts::FieldTimeSeries) = size(fts, 4)
Base.firstindex(fts::FieldTimeSeries) = 1
Base.length(fts::FieldTimeSeries) = size(fts, 4)

# Linear time interpolation
function Base.getindex(fts::FieldTimeSeries, time::Float64)
    Ntimes = length(fts.times)
    t₁, t₂ = index_binary_search(fts.times, time, Ntimes)
    # fractional index
    @inbounds t = (t₂ - t₁) / (fts.times[t₂] - fts.times[t₁]) * (time - fts.times[t₁]) + t₁
    return compute!(Field(fts[t₂] * (t - t₁) + fts[t₁] * (t₂ - t)))
end

# Linear time interpolation
function Base.getindex(fts::FieldTimeSeries, i::Int, j::Int, k::Int, time::Float64)
    Ntimes = length(fts.times)
    t₁, t₂ = index_binary_search(fts.times, time, Ntimes)
    # fractional index
    @inbounds t = (t₂ - t₁) / (fts.times[t₂] - fts.times[t₁]) * (time - fts.times[t₁]) + t₁
    return getindex(fts, i, j, k, t₂) * (t - t₁) + getindex(fts, i,  j, k, t₁) * (t₂ - t)
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
    Nt = length(fts)

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
@propagate_inbounds Base.setindex!(fts::FieldTimeSeries, val, inds...) = Base.setindex!(fts.data, val, inds...)

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
            for n = 1:length(rts)
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

backend_str(::Type{InMemory}) = "InMemory"
backend_str(::Type{OnDisk})   = "OnDisk"
backend_str(::Type{Chunked})  = "Chunked"

#####
##### show
#####

function Base.summary(fts::FieldTimeSeries{LX, LY, LZ, K}) where {LX, LY, LZ, K}
    arch = architecture(fts)
    A = typeof(arch)
    return string("$(join(size(fts), "×")) FieldTimeSeries{$(backend_str(K))} located at ", show_location(fts), " on ", A)
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
           "    └── ", data_summary(fts.data), "\n")

field_time_series_suffix(fts::ChunkedFieldTimeSeries) =
    string("└── data: ", summary(fts.data.data_in_memory), "\n",
            "    └── ", data_summary(fts.data.data_in_memory), "\n")

field_time_series_suffix(fts::OnDiskFieldTimeSeries) =
    string("└── data: ", summary(fts.data))
