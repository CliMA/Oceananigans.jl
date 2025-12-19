using OffsetArrays: OffsetArray
using Oceananigans.Utils
using Oceananigans.Grids: architecture, halo_size

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

const NoBCs = Union{Nothing, Missing, Tuple{Vararg{Nothing}}}

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
##### Missing BCS
#####

@inline _fill_west_halo!(j, k, grid, c, ::Missing, args...)   = nothing
@inline _fill_east_halo!(j, k, grid, c, ::Missing, args...)   = nothing
@inline _fill_south_halo!(i, k, grid, c, ::Missing, args...)  = nothing
@inline _fill_north_halo!(i, k, grid, c, ::Missing, args...)  = nothing
@inline _fill_bottom_halo!(i, j, grid, c, ::Missing, args...) = nothing
@inline _fill_top_halo!(i, j, grid, c, ::Missing, args...)    = nothing

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

@inline fill_halo_size(c::OffsetArray, ::WEB, idx, bc, loc, grid) = size(c, 2), size(c, 3)
@inline fill_halo_size(c::OffsetArray, ::SNB, idx, bc, loc, grid) = size(c, 1), size(c, 3)
@inline fill_halo_size(c::OffsetArray, ::TBB, idx, bc, loc, grid) = size(c, 1), size(c, 2)
    
# The offsets are non-zero only if the indices are not Colon
@inline fill_halo_offset(::Symbol, args...) = (0, 0)

@inline function fill_halo_offset(::Tuple, ::WEB, grid, idx) 
    Hx, Hy, Hz = halo_size(grid)

    Oy = idx[2] == Colon ? - Hy : first(idx[2])-1
    Oz = idx[3] == Colon ? - Hz : first(idx[3])-1

    return (Oy, Oz)
end

@inline function fill_halo_offset(::Tuple, ::SNB, grid, idx) 
    Hx, Hy, Hz = halo_size(grid)

    Ox = idx[1] == Colon ? - Hx : first(idx[1])-1
    Oz = idx[3] == Colon ? - Hz : first(idx[3])-1

    return (Ox, Oz)
end

@inline function fill_halo_offset(::Tuple, ::TBB, grid, idx) 
    Hx, Hy, Hz = halo_size(grid)

    Ox = idx[1] == Colon ? - Hx : first(idx[1])-1
    Oy = idx[2] == Colon ? - Hy : first(idx[2])-1

    return (Ox, Oy)
end
