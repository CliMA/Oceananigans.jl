using Base: @propagate_inbounds

using OffsetArrays
using Statistics
using JLD2
using Adapt

using Dates: AbstractTime
using KernelAbstractions: @kernel, @index

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields

using Oceananigans.Grids: topology, total_size, interior_parent_indices, parent_index_range

using Oceananigans.Fields: show_location, interior_view_indices, data_summary, reduced_location,
                           index_binary_search, indices_summary, boundary_conditions

using Oceananigans.Units: Time
using Oceananigans.Utils: launch!

import Oceananigans.Architectures: architecture
import Oceananigans.BoundaryConditions: fill_halo_regions!, BoundaryCondition, getbc
import Oceananigans.Fields: Field, set!, interior, indices, interpolate!

#####
##### Data backends for FieldTimeSeries
#####

abstract type AbstractDataBackend end

struct InMemory{S} <: AbstractDataBackend 
    start :: S
    size :: S
end

"""
    InMemory(size=nothing)

Return a `backend` for `FieldTimeSeries` that stores `size`
fields in memory. The default `size = nothing` stores all fields in memory.
"""
function InMemory(size::Int)
    size < 2 && throw(ArgumentError("InMemory `size` must be 2 or greater."))
    return InMemory(1, size)
end

InMemory() = InMemory(nothing, nothing)

const TotallyInMemory = InMemory{Nothing}
const  PartlyInMemory = InMemory{Int}

"""
    OnDisk()

Return a lazy `backend` for `FieldTimeSeries` that keeps data
on disk, only loading it as requested by indexing into the
`FieldTimeSeries`.
"""
struct OnDisk <: AbstractDataBackend end

#####
##### Time indexing modes for FieldTimeSeries
#####

"""
    Cyclical(period=nothing)

Specifies cyclical FieldTimeSeries linear Time extrapolation. If
`period` is not specified, it is inferred from the `fts::FieldTimeSeries` via

```julia
t = fts.times
Δt = t[end] - t[end-1]
period = t[end] - t[1] + Δt
```
"""
struct Cyclical{FT}
    period :: FT
end 

Cyclical() = Cyclical(nothing)

"""
    Linear()

Specifies FieldTimeSeries linear Time extrapolation.
"""
struct Linear end

"""
    Clamp()

Specifies FieldTimeSeries Time extrapolation that returns data from the nearest value.
"""
struct Clamp end # clamp to nearest value

# Return the memory index associated with time index `n`.
# For example, if all data is in memory this is simply `n`.
# If only part of the data is in memory, then this is `n - n₀ + 1`,
# where n₀ is the time-index of the first memory index.
#
# TODO: do we need a special `memory_index` implementation for `Clamp` as well?
# For example, it may be better to get a bounds error.

@inline shift_index(n, n₀) = n - n₀ + 1
@inline memory_index(backend, ti, Nt, n) = n
@inline memory_index(backend::TotallyInMemory, ::Cyclical, Nt, n) = mod1(n, Nt)
@inline memory_index(backend::TotallyInMemory, ::Clamp, Nt, n)    = clamp(n, 1, Nt)

@inline memory_index(backend::PartlyInMemory, ::Linear, Nt, n) = shift_index(n, backend.start)

@inline function memory_index(backend::PartlyInMemory, ::Clamp, Nt, n)
    n = clamp(n, 1, Nt)
    return shift_index(n, backend.start)
end

@inline function memory_index(backend::PartlyInMemory, ::Cyclical, Nt, n)
    m = shift_index(n, backend.start)
    m̃ = mod1(m, Nt) # wrap index
    return m̃
end

"""
    time_indices_in_memory(backend::PartlyInMemory, indexing, times)

Return a collection of the time indices that are currently in memory.
This is the inverse of `memory_index`, which converts time indices
to memory indices.
"""
function time_indices_in_memory(backend::PartlyInMemory, ti, times)
    Nt = length(times)
    St = backend.size
    n₀ = backend.start

    time_indices = ntuple(St) do m
        nn = m + n₀ - 1
        mod1(nn, Nt)
    end

    return time_indices
end

time_indices_in_memory(::TotallyInMemory, ti, times) = 1:length(times)

#####
##### FieldTimeSeries
#####

mutable struct FieldTimeSeries{LX, LY, LZ, TI, K, I, D, G, ET, B, χ, P, N} <: AbstractField{LX, LY, LZ, G, ET, 4}
                   data :: D
                   grid :: G
                backend :: K
    boundary_conditions :: B
                indices :: I
                  times :: χ
                   path :: P
                   name :: N
          time_indexing :: TI
    
    function FieldTimeSeries{LX, LY, LZ}(data::D,
                                         grid::G,
                                         backend::K,
                                         bcs::B,
                                         indices::I, 
                                         times,
                                         path,
                                         name,
                                         time_indexing) where {LX, LY, LZ, K, D, G, B, I}

        ET = eltype(data)

        # Check consistency between `backend` and `times`.
        if backend isa PartlyInMemory && backend.size > length(times)
            throw(ArgumentError("`backend.size` cannot be greater than `length(times)`."))
        end

        if times isa AbstractArray
            # Try to convert to a range, cuz
            time_range = range(first(times), last(times), length=length(times))
            if all(time_range .≈ times) # good enough for most
                times = time_range
            end

            times = arch_array(architecture(grid), times)
        end
        
        if time_indexing isa Cyclical{Nothing} # we have to infer the period
            Δt = times[end] - times[end-1]
            period = times[end] - times[1] + Δt
            time_indexing = Cyclical(period)
        end

        χ = typeof(times)
        TI = typeof(time_indexing)
        P = typeof(path)
        N = typeof(name)

        return new{LX, LY, LZ, TI, K, I, D, G, ET, B, χ, P, N}(data, grid, backend, bcs,
                                                               indices, times, path, name,
                                                               time_indexing)
    end
end

#####
##### Minimal implementation of FieldTimeSeries for use in GPU kernels
#####
##### Supports reduced locations + time-interpolation / extrapolation
#####

struct GPUAdaptedFieldTimeSeries{LX, LY, LZ, TI, K, ET, D, χ} <: AbstractArray{ET, 4}
             data :: D
            times :: χ
          backend :: K
    time_indexing :: TI
    
    function GPUAdaptedFieldTimeSeries{LX, LY, LZ}(data::D,
                                                   times::χ,
                                                   backend::K,
                                                   time_indexing::TI) where {LX, LY, LZ, TI, K, D, χ}

        ET = eltype(fts.data)
        return new{LX, LY, LZ, TI, K, ET, D, χ}(data, times, backend, time_indexing)
    end
end

function Adapt.adapt_structure(to, fts::FieldTimeSeries)
    LX, LY, LZ = location(fts)
    return GPUAdaptedFieldTimeSeries{LX, LY, LZ}(adapt(to, fts.data),
                                                 adapt(to, fts.times),
                                                 adapt(to, fts.backend),
                                                 adapt(to, fts.time_indexing))
end

const    FTS{LX, LY, LZ, TI, K} =           FieldTimeSeries{LX, LY, LZ, TI, K} where {LX, LY, LZ, TI, K}
const GPUFTS{LX, LY, LZ, TI, K} = GPUAdaptedFieldTimeSeries{LX, LY, LZ, TI, K} where {LX, LY, LZ, TI, K}

const FlavorOfFTS{LX, LY, LZ, TI, K} = Union{GPUFTS{LX, LY, LZ, TI, K},
                                                FTS{LX, LY, LZ, TI, K}} where {LX, LY, LZ, TI, K} 

const InMemoryFTS        = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:InMemory}
const OnDiskFTS          = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:OnDisk}
const TotallyInMemoryFTS = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:TotallyInMemory}
const PartlyInMemoryFTS  = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:PartlyInMemory}

const CyclicalFTS{K} = FlavorOfFTS{<:Any, <:Any, <:Any, <:Cyclical, K} where K
const   LinearFTS{K} = FlavorOfFTS{<:Any, <:Any, <:Any, <:Linear,   K} where K
const    ClampFTS{K} = FlavorOfFTS{<:Any, <:Any, <:Any, <:Clamp,    K} where K

const CyclicalChunkedFTS = CyclicalFTS{<:PartlyInMemory}

architecture(fts::FieldTimeSeries) = architecture(fts.grid)
time_indices_in_memory(fts) = time_indices_in_memory(fts.backend, fts.time_indexing, fts.times)

@inline function memory_index(fts, n)
    backend = fts.backend
    ti = fts.time_indexing
    Nt = length(fts.times)
    return memory_index(backend, ti, Nt, n)
end

#####
##### Constructors
#####

instantiate(T::Type) = T()

new_data(FT, grid, loc, indices, ::Nothing) = nothing

function new_data(FT, grid, loc, indices, Nt::Int)
    space_size = total_size(grid, loc, indices)
    underlying_data = zeros(FT, architecture(grid), space_size..., Nt)
    data = offset_data(underlying_data, grid, loc, indices)
    return data
end

time_indices_size(backend, times) = throw(ArgumentError("$backend is not a supported backend!"))
time_indices_size(::TotallyInMemory, times) = length(times)
time_indices_size(backend::PartlyInMemory, times) = backend.size
time_indices_size(::OnDisk, times) = nothing

function FieldTimeSeries(loc, grid, times=();
                         indices = (:, :, :), 
                         backend = InMemory(),
                         path = nothing, 
                         name = nothing,
                         time_indexing = Linear(),
                         boundary_conditions = nothing)

    LX, LY, LZ = loc

    Nt = time_indices_size(backend, times)
    data = new_data(eltype(grid), grid, loc, indices, Nt)

    if backend isa OnDisk
        isnothing(name) && isnothing(name) &&
            error(ArgumentError("Must provide the keyword arguments `path` and `name` when `backend=OnDisk()`."))

        isnothing(path) && error(ArgumentError("Must provide the keyword argument `path` when `backend=OnDisk()`."))
        isnothing(name) && error(ArgumentError("Must provide the keyword argument `name` when `backend=OnDisk()`."))
    end
    
    return FieldTimeSeries{LX, LY, LZ}(data, grid, backend, boundary_conditions,
                                       indices, times, path, name, time_indexing)
end

"""
    FieldTimeSeries{LX, LY, LZ}(grid::AbstractGrid, times=(); kwargs...)

Construct a `FieldTimeSeries` on `grid` and at `times`.

Keyword arguments
=================

- `indices`: spatial indices

- `backend`: backend, `InMemory(indices=Colon())` or `OnDisk()`

- `path`: path to data for `backend = OnDisk()`

- `name`: name of field for `backend = OnDisk()`
"""
function FieldTimeSeries{LX, LY, LZ}(grid::AbstractGrid, times=(); kwargs...) where {LX, LY, LZ}
    loc = (LX, LY, LZ)
    return FieldTimeSeries(loc, grid, times; kwargs...)
end

"""
    FieldTimeSeries(path, name, backend = InMemory();
                    grid = nothing,
                    iterations = nothing,
                    times = nothing)

Return a `FieldTimeSeries` containing a time-series of the field `name`
load from JLD2 output located at `path`.

Keyword arguments
=================

- `backend`: `InMemory()` to load data into a 4D array, `OnDisk()` to lazily load data from disk
             when indexing into `FieldTimeSeries`.

- `grid`: A grid to associate with the data, in the case that the native grid was not serialized
          properly.

- `iterations`: Iterations to load. Defaults to all iterations found in the file.

- `times`: Save times to load, as determined through an approximate floating point
           comparison to recorded save times. Defaults to times associated with `iterations`.
           Takes precedence over `iterations` if `times` is specified.
"""
FieldTimeSeries(path::String, name::String; backend=InMemory(), kw...) =
    FieldTimeSeries(path, name, backend; kw...)

struct UnspecifiedBoundaryConditions end

function FieldTimeSeries(path::String, name::String, backend::AbstractDataBackend;
                         architecture = nothing,
                         grid = nothing,
                         location = nothing,
                         boundary_conditions = UnspecifiedBoundaryConditions(),
                         time_indexing = Linear(),
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

    if isnothing(architecture) # determine architecture
        if isnothing(grid) # go to default
            architecture = CPU()
        else # there's a grid, use that architecture
            architecture = Architectures.architecture(grid)
        end
    end

    # This should be removed eventually... (4/5/2022)
    grid = try
        on_architecture(architecture, grid)
    catch err # Likely, the grid was saved with CuArrays or generated with a different Julia version.
        if grid isa RectilinearGrid # we can try...
            @info "Initial attempt to transfer grid to $architecture failed."
            @info "Attempting to reconstruct RectilinearGrid on $architecture manually..."

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

            N = (Nx, Ny, Nz)

            # Reduce for Flat dimensions
            domain = Dict()
            for (i, ξ) in enumerate((x, y, z))
                if topo[i] !== Flat
                    if !(ξ isa Tuple)
                        chopped_ξ = ξ[1:N[i]+1]
                    else
                        chopped_ξ = ξ
                    end
                    sξ = (:x, :y, :z)[i]
                    domain[sξ] = chopped_ξ
                end
            end

            size = Tuple(N[i] for i=1:3 if topo[i] !== Flat)
            halo = Tuple((Hx, Hy, Hz)[i] for i=1:3 if topo[i] !== Flat)

            RectilinearGrid(architecture; size, halo, topology=topo, domain...)
        else
            throw(err)
        end
    end

    close(file)

    LX, LY, LZ = Location

    loc = map(instantiate, Location)
    Nt = time_indices_size(backend, times)
    data = new_data(eltype(grid), grid, loc, indices, Nt)

    time_series = FieldTimeSeries{LX, LY, LZ}(data, grid, backend, boundary_conditions,
                                              indices, times, path, name, time_indexing)

    set!(time_series, path, name)

    return time_series
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

    # Default to CPU if neither architecture nor grid is specified
    if isnothing(architecture)
        if isnothing(grid)
            architecture = CPU()
        else
            architecture = Architectures.architecture(grid)
        end
    end
    
    # Load the grid and data from file
    file = jldopen(path)

    isnothing(grid) && (grid = file["serialized/grid"])
    raw_data = file["timeseries/$name/$iter"]

    close(file)

    # Change grid to specified architecture?
    grid     = on_architecture(architecture, grid)
    raw_data = arch_array(architecture, raw_data)
    data     = offset_data(raw_data, grid, location, indices)
    
    return Field(location, grid; boundary_conditions, indices, data)
end

#####
##### Basic behavior
#####

Base.lastindex(fts::FlavorOfFTS, dim) = lastindex(fts.data, dim)
Base.parent(fts::InMemoryFTS)         = parent(fts.data)
Base.parent(fts::OnDiskFTS)           = nothing
indices(fts::FieldTimeSeries)         = fts.indices
interior(fts::FieldTimeSeries, I...)  = view(interior(fts), I...)

# Make FieldTimeSeries behave like Vector wrt to singleton indices
Base.length(fts::FlavorOfFTS)     = length(fts.times)
Base.lastindex(fts::FlavorOfFTS)  = length(fts.times)
Base.firstindex(fts::FlavorOfFTS) = 1

Base.length(fts::PartlyInMemoryFTS) = fts.backend.size

function interior(fts::FieldTimeSeries)
    loc = map(instantiate, location(fts))
    topo = map(instantiate, topology(fts.grid))
    sz = size(fts.grid)
    halo_sz = halo_size(fts.grid)

    i_interior = map(interior_parent_indices, loc, topo, sz, halo_sz)
    indices = fts.indices
    i_view = map(interior_view_indices, indices, i_interior)

    return view(parent(fts), i_view..., :)
end

# FieldTimeSeries boundary conditions
const CPUFTSBC = BoundaryCondition{<:Any, <:FieldTimeSeries}
const GPUFTSBC = BoundaryCondition{<:Any, <:GPUAdaptedFieldTimeSeries}
const FTSBC = Union{CPUFTSBC, GPUFTSBC}

@inline getbc(bc::FTSBC, i::Int, j::Int, grid::AbstractGrid, clock, args...) = bc.condition[i, j, Time(clock.time)]

#####
##### Fill halo regions
#####

const MAX_FTS_TUPLE_SIZE = 10

fill_halo_regions!(fts::OnDiskFTS) = nothing

function fill_halo_regions!(fts::InMemoryFTS)
    partitioned_indices = Iterators.partition(time_indices_in_memory(fts), MAX_FTS_TUPLE_SIZE)
    partitioned_indices = collect(partitioned_indices)
    Ni = length(partitioned_indices)

    asyncmap(1:Ni) do i
        indices = partitioned_indices[i]
        fts_tuple = Tuple(fts[n] for n in indices)
        fill_halo_regions!(fts_tuple)
    end

    return nothing
end

