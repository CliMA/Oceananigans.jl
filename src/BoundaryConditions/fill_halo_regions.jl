using OffsetArrays: OffsetArray
using Oceananigans.Utils
using Oceananigans.Grids: architecture

import Base

#####
##### General halo filling functions
#####

fill_halo_regions!(::Ref, args...; kwargs...) = nothing # a lot of Refs are passed around, so we need this
fill_halo_regions!(::Nothing, args...; kwargs...) = nothing

"""
    fill_halo_regions!(fields::Union{Tuple, NamedTuple}, arch, args...)

Fill halo regions for each field in the tuple `fields` according to their boundary
conditions, possibly recursing into `fields` if it is a nested tuple-of-tuples.
"""
# Some fields have `nothing` boundary conditions, such as `FunctionField` and `ZeroField`.
fill_halo_regions!(c::OffsetArray, ::Nothing, args...; kwargs...) = nothing


"Fill halo regions in ``x``, ``y``, and ``z`` for a given field's data."
function fill_halo_regions!(c::OffsetArray, boundary_conditions, indices, loc, grid, args...; kwargs...)
    
    kernels!, bcs = get_boundary_kernels(boundary_conditions, c, grid, loc, indices)
    number_of_tasks = length(kernels!)

    # Fill halo in the three permuted directions (1, 2, and 3), making sure dependencies are fulfilled
    for task = 1:number_of_tasks
        @inbounds fill_halo_event!(c, kernels![task], bcs[task], loc, grid, args...; kwargs...)
    end

    return nothing
end

const NoBCs = Union{Nothing, Tuple{Vararg{Nothing}}}

@inline fill_halo_event!(c, kernel!, bcs, loc, grid, args...; kwargs...) = kernel!(c, bcs..., loc, grid, Tuple(args))
@inline fill_halo_event!(c, ::Nothing, ::NoBCs, loc, grid, args...; kwargs...) = nothing

#####
##### Nothing BCs
#####

@inline _fill_west_halo!(j, k, grid, c, ::Nothing, args...)   = nothing
@inline _fill_east_halo!(j, k, grid, c, ::Nothing, args...)   = nothing
@inline _fill_south_halo!(i, k, grid, c, ::Nothing, args...)  = nothing
@inline _fill_north_halo!(i, k, grid, c, ::Nothing, args...)  = nothing
@inline _fill_bottom_halo!(i, j, grid, c, ::Nothing, args...) = nothing
@inline _fill_top_halo!(i, j, grid, c, ::Nothing, args...)    = nothing

#####
##### Double-sided fill_halo! kernels
#####

@kernel function _fill_west_and_east_halo!(c, west_bc, east_bc, loc, grid, args)
    j, k = @index(Global, NTuple)
    _fill_west_halo!(j, k, grid, c, west_bc, loc, args...)
    _fill_east_halo!(j, k, grid, c, east_bc, loc, args...)
end

@kernel function _fill_south_and_north_halo!(c, south_bc, north_bc, loc, grid, args)
    i, k = @index(Global, NTuple)
    _fill_south_halo!(i, k, grid, c, south_bc, loc, args...)
    _fill_north_halo!(i, k, grid, c, north_bc, loc, args...)
end

@kernel function _fill_bottom_and_top_halo!(c, bottom_bc, top_bc, loc, grid, args)
    i, j = @index(Global, NTuple)
    _fill_bottom_halo!(i, j, grid, c, bottom_bc, loc, args...)
       _fill_top_halo!(i, j, grid, c, top_bc,    loc, args...)
end

#####
##### Single-sided fill_halo! kernels
#####

@kernel function _fill_only_west_halo!(c, bc, loc, grid, args)
    j, k = @index(Global, NTuple)
    _fill_west_halo!(j, k, grid, c, bc, loc, args...)
end

@kernel function _fill_only_south_halo!(c, bc, loc, grid, args)
    i, k = @index(Global, NTuple)
    _fill_south_halo!(i, k, grid, c, bc, loc, args...)
end

@kernel function _fill_only_bottom_halo!(c, bc, loc, grid, args)
    i, j = @index(Global, NTuple)
    _fill_bottom_halo!(i, j, grid, c, bc, loc, args...)
end

@kernel function _fill_only_east_halo!(c, bc, loc, grid, args)
    j, k = @index(Global, NTuple)
    _fill_east_halo!(j, k, grid, c, bc, loc, args...)
end

@kernel function _fill_only_north_halo!(c, bc, loc, grid, args)
    i, k = @index(Global, NTuple)
    _fill_north_halo!(i, k, grid, c, bc, loc, args...)
end

@kernel function _fill_only_top_halo!(c, bc, loc, grid, args)
    i, j = @index(Global, NTuple)
    _fill_top_halo!(i, j, grid, c, bc, loc, args...)
end

#####
##### Calculate kernel size and offset for Windowed and Sliced Fields
#####

const WEB = Union{WestAndEast, West, East}
const SNB = Union{SouthAndNorth, South, North}
const TBB = Union{BottomAndTop, Bottom, Top}

# Tupled halo filling _only_ deals with full fields!
@inline fill_halo_size(::Tuple, ::WEB, args...) = :yz
@inline fill_halo_size(::Tuple, ::SNB, args...) = :xz
@inline fill_halo_size(::Tuple, ::TBB, args...) = :xy

# If indices are colon, and locations are _not_ Nothing, fill the whole boundary plane!
# If locations are _Nothing_, then the kwarg `reduced_dimensions` will allow the size `:xz`
# to be correctly interpreted inside `launch!`.
@inline fill_halo_size(::OffsetArray, ::WEB, ::Tuple{<:Any, <:Colon, <:Colon}, args...) = :yz
@inline fill_halo_size(::OffsetArray, ::SNB, ::Tuple{<:Colon, <:Any, <:Colon}, args...) = :xz
@inline fill_halo_size(::OffsetArray, ::TBB, ::Tuple{<:Colon, <:Colon, <:Any}, args...) = :xy

# If the index is a Colon and the location is _NOT_ a `Nothing` (i.e. not a `ReducedField`),
# then fill the whole boundary, otherwise fill the size of the corresponding array
@inline whole_halo(idx, loc)           = false
@inline whole_halo(idx,     ::Nothing) = false
@inline whole_halo(::Colon, ::Nothing) = false
@inline whole_halo(::Colon,       loc) = true

# Calculate kernel size for windowed fields. This code is only called when
# one or more of the elements of `idx` is not Colon in the two direction perpendicular
# to the halo region and `bc` is not `PeriodicBoundaryCondition`.
@inline function fill_halo_size(c::OffsetArray, ::WEB, idx, bc, loc, grid)
    @inbounds begin
        whole_y_halo = whole_halo(idx[2], loc[2])
        whole_z_halo = whole_halo(idx[3], loc[3])
    end

    _, Ny, Nz = size(grid)
    _, Cy, Cz = size(c)

    Sy = ifelse(whole_y_halo, Ny, Cy)
    Sz = ifelse(whole_z_halo, Nz, Cz)

    return (Sy, Sz)
end

@inline function fill_halo_size(c::OffsetArray, ::SNB, idx, bc, loc, grid)
    @inbounds begin
        whole_x_halo = whole_halo(idx[1], loc[1])
        whole_z_halo = whole_halo(idx[3], loc[3])
    end

    Nx, _, Nz = size(grid)
    Cx, _, Cz = size(c)

    Sx = ifelse(whole_x_halo, Nx, Cx)
    Sz = ifelse(whole_z_halo, Nz, Cz)

    return (Sx, Sz)
end

@inline function fill_halo_size(c::OffsetArray, ::TBB, idx, bc, loc, grid)
    @inbounds begin
        whole_x_halo = whole_halo(idx[1], loc[1])
        whole_y_halo = whole_halo(idx[2], loc[2])
    end

    Nx, Ny, _ = size(grid)
    Cx, Cy, _ = size(c)

    Sx = ifelse(whole_x_halo, Nx, Cx)
    Sy = ifelse(whole_y_halo, Ny, Cy)

    return (Sx, Sy)
end

# Remember that Periodic BCs also fill halo points!
@inline fill_halo_size(c::OffsetArray, ::WEB, idx, ::PBC, args...) = tuple(size(c, 2), size(c, 3))
@inline fill_halo_size(c::OffsetArray, ::SNB, idx, ::PBC, args...) = tuple(size(c, 1), size(c, 3))
@inline fill_halo_size(c::OffsetArray, ::TBB, idx, ::PBC, args...) = tuple(size(c, 1), size(c, 2))

@inline function fill_halo_size(c::OffsetArray, ::WEB, ::Tuple{<:Any, <:Colon, <:Colon}, ::PBC, args...)
    _, Cy, Cz = size(c)
    return (Cy, Cz)
end

@inline function fill_halo_size(c::OffsetArray, ::SNB, ::Tuple{<:Colon, <:Any, <:Colon}, ::PBC, args...)
    Cx, _, Cz = size(c)
    return (Cx, Cz)
end

@inline function fill_halo_size(c::OffsetArray, ::TBB, ::Tuple{<:Colon, <:Colon, <:Any}, ::PBC, args...)
    Cx, Cy, _ = size(c)
    return (Cx, Cy)
end

# The offsets are non-zero only if the indices are not Colon
@inline fill_halo_offset(::Symbol, args...)   = (0, 0)
@inline fill_halo_offset(::Tuple, ::WEB, idx) = (idx[2] == Colon() ? 0 : first(idx[2])-1, idx[3] == Colon() ? 0 : first(idx[3])-1)
@inline fill_halo_offset(::Tuple, ::SNB, idx) = (idx[1] == Colon() ? 0 : first(idx[1])-1, idx[3] == Colon() ? 0 : first(idx[3])-1)
@inline fill_halo_offset(::Tuple, ::TBB, idx) = (idx[1] == Colon() ? 0 : first(idx[1])-1, idx[2] == Colon() ? 0 : first(idx[2])-1)

