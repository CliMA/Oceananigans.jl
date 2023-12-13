using Base: @propagate_inbounds

using OffsetArrays
using Statistics
using JLD2

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Fields

using Oceananigans.Grids: topology, total_size, interior_parent_indices, parent_index_range
using Oceananigans.Fields: show_location, interior_view_indices, data_summary, reduced_location, index_binary_search,
                           indices_summary, boundary_conditions 

using Oceananigans.Units: Time
using Oceananigans.Utils: launch!

import Oceananigans.Architectures: architecture
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Fields: Field, set!, interior, indices, interpolate!

using Dates: AbstractTime
using KernelAbstractions: @kernel, @index

struct FieldTimeSeries{LX, LY, LZ, K, I, D, G, T, B, χ, P, N} <: AbstractField{LX, LY, LZ, G, T, 4}
                   data :: D
                   grid :: G
                backend :: K
    boundary_conditions :: B
                indices :: I
                  times :: χ
                   path :: P
                   name :: N

    function FieldTimeSeries{LX, LY, LZ}(data::D,
                                         grid::G,
                                         backend::K,
                                         bcs::B,
                                         indices::I, 
                                         times::χ,
                                         path::P,
                                         name::N) where {LX, LY, LZ, K, D, G, B, χ, I, P, N}
        T = eltype(data)
        return new{LX, LY, LZ, K, I, D, G, T, B, χ, P, N}(data, grid, backend, bcs,
                                                          indices, times, path, name)
    end
end

architecture(fts::FieldTimeSeries) = architecture(fts.grid)

const TotallyInMemoryFieldTimeSeries{LX, LY, LZ} = FieldTimeSeries{LX, LY, LZ, <:InMemory{Colon}}
const InMemoryFieldTimeSeries{LX, LY, LZ} = FieldTimeSeries{LX, LY, LZ, <:InMemory}
const OnDiskFieldTimeSeries{LX, LY, LZ} = FieldTimeSeries{LX, LY, LZ, <:OnDisk}

struct UnspecifiedBoundaryConditions end

#####
##### Constructors
#####

instantiate(T::Type) = T()

function FieldTimeSeries(loc, grid, times;
                         indices = (:, :, :), 
                         backend = InMemory(),
                         path = nothing, 
                         name = nothing,
                         boundary_conditions = nothing)

    LX, LY, LZ = loc
    Nt   = length(times)
    data = new_data(eltype(grid), grid, loc, indices, Nt, backend)

    if backend isa OnDisk
        isnothing(name) && isnothing(name) &&
            error(ArgumentError("Must provide the keyword arguments `path` and `name` when `backend=OnDisk()`."))

        isnothing(path) && error(ArgumentError("Must provide the keyword argument `path` when `backend=OnDisk()`."))
        isnothing(name) && error(ArgumentError("Must provide the keyword argument `name` when `backend=OnDisk()`."))
    end

    return FieldTimeSeries{LX, LY, LZ}(data, grid, backend, boundary_conditions,
                                       indices, times, path, name)
end

"""
    FieldTimeSeries{LX, LY, LZ}(grid::AbstractGrid, times; kwargs...) where {LX, LY, LZ} =

Construct a `FieldTimeSeries` on `grid` and at `times`.

Keyword arguments
=================

- indices: spatial indices
- backend: backend, `InMemory(indices=Colon())` or `OnDisk()`
- path: path to data for `backend = OnDisk()`
- name: name of field for `backend = OnDisk()`
"""
FieldTimeSeries{LX, LY, LZ}(grid::AbstractGrid, times; kwargs...) where {LX, LY, LZ} =
    FieldTimeSeries((LX, LY, LZ), grid, times; kwargs...)

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

function FieldTimeSeries(path::String, name::String, backend::AbstractDataBackend;
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

    # Default to CPU if neither architecture nor grid is specified
    architecture = isnothing(architecture) ?
        (isnothing(grid) ? CPU() : Architectures.architecture(grid)) : architecture

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
                                              indices, times, path, name)

    set!(time_series, path, name)

    return time_series
end

# Making FieldTimeSeries behave like Vector
Base.lastindex(fts::FieldTimeSeries) = size(fts, 4)
Base.firstindex(fts::FieldTimeSeries) = 1
Base.length(fts::FieldTimeSeries) = size(fts, 4)

# Linear time interpolation
function Base.getindex(fts::FieldTimeSeries, time_index::Time)
    Ntimes = length(fts.times)
    time = time_index.time
    n₁, n₂ = index_binary_search(fts.times, time, Ntimes)
    if n₁ == n₂ # no interpolation
        return fts[n₁]
    end
    
    # Calculate fractional index
    n = @inbounds (n₂ - n₁) / (fts.times[n₂] - fts.times[n₁]) * (time - fts.times[n₁]) + n₁

    # Make a Field representing a linear interpolation in time
    time_interpolated_field = Field(fts[n₂] * (n - n₁) + fts[n₁] * (n₂ - n))

    # Compute the field and return it
    return compute!(time_interpolated_field)
end

# Linear time interpolation, used by FieldTimeSeries and GPUAdaptedFieldTimeSeries
@inline function interpolating_get_index(fts, i, j, k, time_index)
    Ntimes = length(fts.times)
    time = time_index.time
    n₁, n₂ = index_binary_search(fts.times, time, Ntimes)

    # fractional index
    n = @inbounds (n₂ - n₁) / (fts.times[n₂] - fts.times[n₁]) * (time - fts.times[n₁]) + n₁
    interpolated_fts = getindex(fts, i, j, k, n₂) * (n - n₁) + getindex(fts, i, j, k, n₁) * (n₂ - n)

    # Don't interpolate if n = 0.
    return ifelse(n₁ == n₂, getindex(fts, i, j, k, n₁), interpolated_fts)
end

@inline Base.getindex(fts::FieldTimeSeries, i::Int, j::Int, k::Int, time_index::Time) =
    interpolating_get_index(fts, i, j, k, time_index)

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

#####
##### set!
#####

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

#####
##### show
#####

function Base.summary(fts::FieldTimeSeries{LX, LY, LZ, K}) where {LX, LY, LZ, K}
    arch = architecture(fts)
    B = string(typeof(fts.backend).name.wrapper)
    sz_str = string(join(size(fts), "×"))

    path = fts.path
    name = fts.name
    A = typeof(arch)

    if isnothing(path)
        suffix = " on $A"
    else
        suffix = " of $name at $path"
    end

    return string("$sz_str FieldTimeSeries{$B} located at ", show_location(fts), suffix)
end

function Base.show(io::IO, fts::FieldTimeSeries)
    prefix = string(summary(fts), '\n',
                   "├── grid: ", summary(fts.grid), '\n',
                   "├── indices: ", indices_summary(fts), '\n')

    suffix = field_time_series_suffix(fts)

    return print(io, prefix, suffix)
end

function field_time_series_suffix(fts::InMemoryFieldTimeSeries)
    ii = fts.backend.index_range

    if ii isa Colon
        backend_str = "├── backend: InMemory(:)"
    else
        N = length(ii)
        if N < 6
            index_range_str = string(ii)
        else
            index_range_str = string("[", ii[1],
                                     ", ", ii[2],
                                     ", ", ii[3],
                                     "  …  ",
                                     ii[end-2], ", ",
                                     ii[end-1], ", ",
                                     ii[end], "]")
                                     
        end

        backend_str = string("├── backend: InMemory(", index_range_str, ")", '\n')
    end

    path_str = isnothing(fts.path) ? "" : string("├── path: ", fts.path, '\n')
    name_str = isnothing(fts.name) ? "" : string("├── name: ", fts.name)

    return string(backend_str, '\n',
                  path_str,
                  name_str,
                  "└── data: ", summary(fts.data), '\n',
                  "    └── ", data_summary(fts.data))
end

field_time_series_suffix(fts::OnDiskFieldTimeSeries) =
    string("├── backend: ", summary(fts.backend), '\n',
           "├── path: ", fts.path, '\n',
           "└── name: ", fts.name)

