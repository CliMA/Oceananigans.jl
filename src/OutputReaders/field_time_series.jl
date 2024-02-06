using Base: @propagate_inbounds

using OffsetArrays
using Statistics
using JLD2

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
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Fields: Field, set!, interior, indices, interpolate!

tupleit(t::Tuple) = t
tupleit(t) = Tuple(t)

mutable struct FieldTimeSeries{LX, LY, LZ, TE, K, I, D, G, T, B, χ, P, N} <: AbstractField{LX, LY, LZ, G, T, 4}
                   data :: D
                   grid :: G
                backend :: K
    boundary_conditions :: B
                indices :: I
                  times :: χ
                   path :: P
                   name :: N
     time_extrapolation :: TE
    
    function FieldTimeSeries{LX, LY, LZ}(data::D,
                                         grid::G,
                                         backend::K,
                                         bcs::B,
                                         indices::I, 
                                         times,
                                         path::P,
                                         name::N,
                                         te::TE) where {LX, LY, LZ, TE, K, D, G, B, I, P, N}

        T = eltype(data)
        # TODO: check for equal spacing and convert to range
        # times = tupleit(times)
        χ = typeof(times)

        # TODO: check for consistency between backend and times.
        # For example, for backend::InMemory, backend.indices cannot havex
        # more entries than times.

        return new{LX, LY, LZ, TE, K, I, D, G, T, B, χ, P, N}(data, grid, backend, bcs,
                                                              indices, times, path, name, te)
    end
end

struct GPUAdaptedFieldTimeSeries{LX, LY, LZ, TE, K, T, D, χ} <: AbstractArray{T, 4}
    data :: D
    times :: χ
    backend :: K
    time_extrapolation :: TE
    
    function GPUAdaptedFieldTimeSeries{LX, LY, LZ, T}(data::D,
                                                      times::χ,
                                                      backend::K,
                                                      time_extrapolation::TE) where {LX, LY, LZ, TE, K, T, D, χ}

        return new{LX, LY, LZ, TE, K, T, D, χ}(data, times, backend, time_extrapolation)
    end
end

function Adapt.adapt_structure(to, fts)
    T = eltype(fts.data)
    LX, LY, LZ = location(fts)
    return GPUAdaptedFieldTimeSeries{LX, LY, LZ, T}(adapt(to, fts.data),
                                                    adapt(to, fts.times),
                                                    adapt(to, fts.backend),
                                                    adapt(to, fts.time_extrapolation))
end

@propagate_inbounds Base.lastindex(fts::GPUAdaptedFieldTimeSeries) = lastindex(fts.data)
@propagate_inbounds Base.lastindex(fts::GPUAdaptedFieldTimeSeries, dim) = lastindex(fts.data, dim)

const    FTS{LX, LY, LZ, TE, K} = FieldTimeSeries{LX, LY, LZ, TE, K} where {LX, LY, LZ, TE, K}
const GPUFTS{LX, LY, LZ, TE, K} = GPUAdaptedFieldTimeSeries{LX, LY, LZ, TE, K} where {LX, LY, LZ, TE, K}

const FlavorOfFTS{LX, LY, LZ, TE, K} = Union{GPUFTS{LX, LY, LZ, TE, K},
                                                FTS{LX, LY, LZ, TE, K}} where {LX, LY, LZ, TE, K} 


const TotallyInMemory = InMemory{Nothing, Nothing}
const  PartlyInMemory = InMemory{Int, Int}

const InMemoryFTS        = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:InMemory}
const TotallyInMemoryFTS = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:TotallyInMemory}
const PartlyInMemoryFTS  = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:PartlyInMemory}

const CyclicalChunkedFTS = CyclicalFTS{<:PartlyInMemory}


architecture(fts::FieldTimeSeries) = architecture(fts.grid)

#####
##### Constructors
#####

instantiate(T::Type) = T()

function FieldTimeSeries(loc, grid, times=();
                         indices = (:, :, :), 
                         backend = InMemory(),
                         path = nothing, 
                         name = nothing,
                         time_extrapolation = Linear(),
                         boundary_conditions = nothing)

    LX, LY, LZ = loc
    Nt = length(times)
    data = new_data(eltype(grid), grid, loc, indices, Nt, backend)

    if backend isa OnDisk
        isnothing(name) && isnothing(name) &&
            error(ArgumentError("Must provide the keyword arguments `path` and `name` when `backend=OnDisk()`."))

        isnothing(path) && error(ArgumentError("Must provide the keyword argument `path` when `backend=OnDisk()`."))
        isnothing(name) && error(ArgumentError("Must provide the keyword argument `name` when `backend=OnDisk()`."))
    end

    if time_extrapolation isa Cyclical{Nothing} # infer the period
        Δt = times[end] - times[end-1]
        period = times[end] - times[1] + Δt
        time_extrapolation = Cyclical(period)
    end

    return FieldTimeSeries{LX, LY, LZ}(data, grid, backend, boundary_conditions,
                                       indices, times, path, name, time_extrapolation)
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
                         time_extrapolation = Linear(),
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
    Nt = length(times)
    data = new_data(eltype(grid), grid, loc, indices, Nt, backend)

    time_series = FieldTimeSeries{LX, LY, LZ}(data, grid, backend, boundary_conditions,
                                              indices, times, path, name, time_extrapolation)

    set!(time_series, path, name)

    return time_series
end

# Making FieldTimeSeries behave like Vector
Base.lastindex(fts::FieldTimeSeries) = size(fts, 4)
Base.firstindex(fts::FieldTimeSeries) = 1
Base.length(fts::FieldTimeSeries) = size(fts, 4)

function interpolate!(target_fts::FieldTimeSeries, source_fts::FieldTimeSeries)

    # TODO: support time-interpolation too.
    # This requires extending the low-level Field interpolation utilities
    # to support time-indexing.
    target_fts.times == source_fts.times ||
        throw(ArgumentError("Cannot interpolate two FieldTimeSeries with different times."))

    times = target_fts.times
    Nt = length(times)

    target_grid = target_fts.grid
    source_grid = source_fts.grid

    @assert architecture(target_grid) == architecture(source_grid)
    arch = architecture(target_grid)

    # Make locations
    source_location = Tuple(L() for L in location(source_fts))
    target_location = Tuple(L() for L in location(target_fts))

    launch!(arch, target_grid, size(target_fts),
            _interpolate_field_time_series!,
            target_fts.data, target_grid, target_location,
            source_fts.data, source_grid, source_location)

    fill_halo_regions!(target_fts)

    return nothing
end

@kernel function _interpolate_field_time_series!(target_fts, target_grid, target_location,
                                                 source_fts, source_grid, source_location)

    # 4D index, cool!
    i, j, k, n = @index(Global, NTuple)

    source_field = view(source_fts, :, :, :, n)
    target_node = node(i, j, k, target_grid, target_location...)

    @inbounds target_fts[i, j, k, n] = interpolate(target_node, source_field, source_location, source_grid)
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

    if isnothing(grid)
        grid = on_architecture(architecture, file["serialized/grid"])
    end
        
    raw_data = arch_array(architecture, file["timeseries/$name/$iter"])

    close(file)

    data = offset_data(raw_data, grid, location, indices)
    
    return Field(location, grid; boundary_conditions, indices, data)
end

#####
##### set!
#####

const FieldsVector = AbstractVector{<:AbstractField}

function set!(fts::FieldTimeSeries, fields_vector::FieldsVector)
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


