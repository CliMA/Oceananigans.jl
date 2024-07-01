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

using Oceananigans.Fields: interior_view_indices, index_binary_search,
                           indices_summary, boundary_conditions

using Oceananigans.Units: Time
using Oceananigans.Utils: launch!

import Oceananigans.Architectures: architecture, on_architecture
import Oceananigans.BoundaryConditions: fill_halo_regions!, BoundaryCondition, getbc
import Oceananigans.Fields: Field, set!, interior, indices, interpolate!

#####
##### Data backends for FieldTimeSeries
#####

abstract type AbstractDataBackend end
abstract type AbstractInMemoryBackend{S} end

struct InMemory{S} <: AbstractInMemoryBackend{S}
    start :: S
    length :: S
end

"""
    InMemory(length=nothing)

Return a `backend` for `FieldTimeSeries` that stores `size`
fields in memory. The default `size = nothing` stores all fields in memory.
"""
function InMemory(length::Int)
    length < 2 && throw(ArgumentError("InMemory `length` must be 2 or greater."))
    return InMemory(1, length)
end

InMemory() = InMemory(nothing, nothing)

const TotallyInMemory = AbstractInMemoryBackend{Nothing}
const  PartlyInMemory = AbstractInMemoryBackend{Int}

Base.summary(backend::PartlyInMemory) = string("InMemory(", backend.start, ", ", length(backend), ")")
Base.summary(backend::TotallyInMemory) = "InMemory()"

new_backend(::InMemory, start, length) = InMemory(start, length)

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

# Totally in memory stuff
@inline   time_index(backend, ti, Nt, m) = m
@inline memory_index(backend, ti, Nt, n) = n
@inline memory_index(backend::TotallyInMemory, ::Cyclical, Nt, n) = mod1(n, Nt)
@inline memory_index(backend::TotallyInMemory, ::Clamp, Nt, n) = clamp(n, 1, Nt)

# Partly in memory stuff
@inline shift_index(n, n₀) = n - (n₀ - 1)
@inline reverse_index(m, n₀) = m + n₀ - 1

@inline memory_index(backend::PartlyInMemory, ::Linear, Nt, n) = shift_index(n, backend.start)

@inline function memory_index(backend::PartlyInMemory, ::Clamp, Nt, n)
    n̂ = clamp(n, 1, Nt)
    m = shift_index(n̂, backend.start)
    return m
end

"""
    time_index(backend::PartlyInMemory, time_indexing, Nt, m)

Compute the time index of a snapshot currently stored at the memory index `m`,
given `backend`, `time_indexing`, and number of times `Nt`.
"""
@inline time_index(backend::PartlyInMemory, ::Union{Clamp, Linear}, Nt, m) =
    reverse_index(m, backend.start)

"""
    memory_index(backend::PartlyInMemory, time_indexing, Nt, n)

Compute the current index of a snapshot in memory that has
the time index `n`, given `backend`, `time_indexing`, and number of times `Nt`.

Example
=======

For `backend::PartlyInMemory` and `time_indexing::Cyclical`:

# Simple shifting example
```julia
Nt = 5
backend = InMemory(2, 3) # so we have (2, 3, 4)
n = 4           # so m̃ = 3
m = 4 - (2 - 1) # = 3
m̃ = mod1(3, 5)  # = 3 ✓
```

# Shifting + wrapping example
```julia
Nt = 5
backend = InMemory(4, 3) # so we have (4, 5, 1)
n = 1 # so, the right answer is m̃ = 3
m = 1 - (4 - 1) # = -2
m̃ = mod1(-2, 5)  # = 3 ✓ 
```

# Another shifting + wrapping example
```julia
Nt = 5
backend = InMemory(5, 3) # so we have (5, 1, 2)
n = 11 # so, the right answer is m̃ = 2
m = 11 - (5 - 1) # = 7
m̃ = mod1(7, 5)  # = 2 ✓
```
"""
@inline function memory_index(backend::PartlyInMemory, ::Cyclical, Nt, n)
    m = shift_index(n, backend.start)
    m̃ = mod1(m, Nt) # wrap index
    return m̃
end

@inline function time_index(backend::PartlyInMemory, ::Cyclical, Nt, m)
    n = reverse_index(m, backend.start)
    ñ = mod1(n, Nt) # wrap index
    return ñ
end

"""
    time_indices(backend, time_indexing, Nt)

Return a collection of the time indices that are currently in memory.
If `backend::TotallyInMemory` then return `1:length(times)`.
"""
function time_indices(backend::PartlyInMemory, time_indexing, Nt)
    St = length(backend)
    n₀ = backend.start

    time_indices = ntuple(St) do m
        time_index(backend, time_indexing, Nt, m)
    end

    return time_indices
end

time_indices(::TotallyInMemory, time_indexing, Nt) = 1:Nt

Base.length(backend::PartlyInMemory) = backend.length

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
        if backend isa PartlyInMemory && backend.length > length(times)
            throw(ArgumentError("`backend.length` cannot be greater than `length(times)`."))
        end

        if times isa AbstractArray
            # Try to convert to a range, cuz
            time_range = range(first(times), last(times), length=length(times))
            if all(time_range .≈ times) # good enough for most
                times = time_range
            end

            times = on_architecture(architecture(grid), times)
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

on_architecture(to, fts::FieldTimeSeries{LX, LY, LZ}) where {LX, LY, LZ} = 
    FieldTimeSeries{LX, LY, LZ}(on_architecture(to, data),
                                on_architecture(to, grid),
                                on_architecture(to, backend),
                                on_architecture(to, bcs),
                                on_architecture(to, indices), 
                                on_architecture(to, times),
                                on_architecture(to, path),
                                on_architecture(to, name),
                                on_architecture(to, time_indexing))

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

        ET = eltype(data)
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

const InMemoryFTS        = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:AbstractInMemoryBackend}
const OnDiskFTS          = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:OnDisk}
const TotallyInMemoryFTS = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:TotallyInMemory}
const PartlyInMemoryFTS  = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:PartlyInMemory}

const CyclicalFTS{K} = FlavorOfFTS{<:Any, <:Any, <:Any, <:Cyclical, K} where K
const   LinearFTS{K} = FlavorOfFTS{<:Any, <:Any, <:Any, <:Linear,   K} where K
const    ClampFTS{K} = FlavorOfFTS{<:Any, <:Any, <:Any, <:Clamp,    K} where K

const CyclicalChunkedFTS = CyclicalFTS{<:PartlyInMemory}

architecture(fts::FieldTimeSeries) = architecture(fts.grid)
time_indices(fts) = time_indices(fts.backend, fts.time_indexing, length(fts.times))

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

# Apparently, not explicitly specifying Int64 in here makes this function
# fail on x86 processors where `Int` is implied to be `Int32` 
# see ClimaOcean commit 3c47d887659d81e0caed6c9df41b7438e1f1cd52 at https://github.com/CliMA/ClimaOcean.jl/actions/runs/8804916198/job/24166354095)
function new_data(FT, grid, loc, indices, Nt::Union{Int, Int64})
    space_size = total_size(grid, loc, indices)
    underlying_data = zeros(FT, architecture(grid), space_size..., Nt)
    data = offset_data(underlying_data, grid, loc, indices)
    return data
end

time_indices_length(backend, times) = throw(ArgumentError("$backend is not a supported backend!"))
time_indices_length(::TotallyInMemory, times) = length(times)
time_indices_length(backend::PartlyInMemory, times) = length(backend)
time_indices_length(::OnDisk, times) = nothing

function FieldTimeSeries(loc, grid, times=();
                         indices = (:, :, :), 
                         backend = InMemory(),
                         path = nothing, 
                         name = nothing,
                         time_indexing = Linear(),
                         boundary_conditions = nothing)

    LX, LY, LZ = loc

    Nt = time_indices_length(backend, times)
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
    FieldTimeSeries{LX, LY, LZ}(grid::AbstractGrid [, times=()]; kwargs...)

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

struct UnspecifiedBoundaryConditions end

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
function FieldTimeSeries(path::String, name::String;
                         backend = InMemory(),
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
    Nt = time_indices_length(backend, times)
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
    raw_data = on_architecture(architecture, raw_data)
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

Base.length(fts::PartlyInMemoryFTS) = length(fts.backend)

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
    partitioned_indices = Iterators.partition(time_indices(fts), MAX_FTS_TUPLE_SIZE)
    partitioned_indices = collect(partitioned_indices)
    Ni = length(partitioned_indices)

    asyncmap(1:Ni) do i
        indices = partitioned_indices[i]
        fts_tuple = Tuple(fts[n] for n in indices)
        fill_halo_regions!(fts_tuple)
    end

    return nothing
end

