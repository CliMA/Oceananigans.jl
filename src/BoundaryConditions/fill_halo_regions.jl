using OffsetArrays: OffsetArray
using Oceananigans.Architectures: device_event
using Oceananigans.Grids: architecture
using KernelAbstractions.Extras.LoopInfo: @unroll

#####
##### General halo filling functions
#####

fill_halo_regions!(::Nothing, args...) = nothing

"""
    fill_halo_regions!(fields::Union{Tuple, NamedTuple}, arch, args...)

Fill halo regions for each field in the tuple `fields` according to their boundary
conditions, possibly recursing into `fields` if it is a nested tuple-of-tuples.
"""

# Some fields have `nothing` boundary conditions, such as `FunctionField` and `ZeroField`.
fill_halo_regions!(c::OffsetArray, ::Nothing, args...; kwargs...) = nothing

for dir in (:west, :east, :south, :north, :bottom, :top)
    extract_bc = Symbol(:extract_, dir, :_bc)
    @eval begin
        @inline $extract_bc(bc) = bc.$dir
        @inline $extract_bc(bc::Tuple) = $extract_bc.(bc)
    end
end

# Finally, the true fill_halo!
"Fill halo regions in ``x``, ``y``, and ``z`` for a given field's data."
function fill_halo_regions!(c::Union{OffsetArray, NTuple{<:Any, OffsetArray}}, boundary_conditions, grid, args...; kwargs...)

    arch = architecture(grid)

    fill_halos! = [
        fill_west_and_east_halo!,
        fill_south_and_north_halo!,
        fill_bottom_and_top_halo!,
    ]

    boundary_conditions_array_left = [
        extract_west_bc(boundary_conditions),
        extract_south_bc(boundary_conditions),
        extract_bottom_bc(boundary_conditions)
    ]

    boundary_conditions_array_right = [
        extract_east_bc(boundary_conditions),
        extract_north_bc(boundary_conditions),
        extract_top_bc(boundary_conditions),
    ]

    perm = sortperm(boundary_conditions_array_left, lt=fill_first)
    fill_halos! = fill_halos![perm]
    boundary_conditions_array_left  = boundary_conditions_array_left[perm]
    boundary_conditions_array_right = boundary_conditions_array_right[perm]
   
    for task = 1:3
        barrier = device_event(arch)

        fill_halo!  = fill_halos![task]
        bc_left     = boundary_conditions_array_left[task]
        bc_right    = boundary_conditions_array_right[task]

        events      = fill_halo!(c, bc_left, bc_right, arch, barrier, grid, args...; kwargs...)
       
        wait(device(arch), events)
    end

    return nothing
end

@inline validate_event(::Nothing) = NoneEvent()
@inline validate_event(event)     = event

#####
##### Halo filling order
#####

const PBCT = Union{PBC, NTuple{<:Any, <:PBC}}

fill_first(bc1::PBCT, bc2)       = false
fill_first(bc1, bc2::PBCT)       = true
fill_first(bc1::PBCT, bc2::PBCT) = true
fill_first(bc1, bc2)             = true

#####
##### General fill_halo! kernels
#####

@kernel function _fill_west_and_east_halo!(c::OffsetArray, west_bc, east_bc, grid, args...)
    j, k = @index(Global, NTuple)
    _fill_west_halo!(j, k, grid, c, west_bc, args...)
    _fill_east_halo!(j, k, grid, c, east_bc, args...)
end

@kernel function _fill_south_and_north_halo!(c::OffsetArray, south_bc, north_bc, grid, args...)
    i, k = @index(Global, NTuple)
    _fill_south_halo!(i, k, grid, c, south_bc, args...)
    _fill_north_halo!(i, k, grid, c, north_bc, args...)
end

@kernel function _fill_bottom_and_top_halo!(c::OffsetArray, bottom_bc, top_bc, grid, args...)
    i, j = @index(Global, NTuple)
    _fill_bottom_halo!(i, j, grid, c, bottom_bc, args...)
       _fill_top_halo!(i, j, grid, c, top_bc, args...)
end

#####
##### Tuple fill_halo! kernels
#####

@kernel function _fill_west_and_east_halo!(c::Tuple, west_bc::Tuple, east_bc::Tuple, grid, args...) 
    j, k = @index(Global, NTuple)
    @unroll for n in 1:length(c)
        _fill_west_halo!(j, k, grid, c[n], west_bc[n], args...)
        _fill_east_halo!(j, k, grid, c[n], east_bc[n], args...)
    end
end

@kernel function _fill_south_and_north_halo!(c::Tuple, south_bc::Tuple, north_bc::Tuple, grid, args...) where N
    i, k = @index(Global, NTuple)
    @unroll for n in 1:length(c)
        _fill_south_halo!(i, k, grid, c[n], south_bc[n], args...)
        _fill_north_halo!(i, k, grid, c[n], north_bc[n], args...)
    end
end

@kernel function _fill_bottom_and_top_halo!(c::Tuple, bottom_bc::Tuple, top_bc::Tuple, grid, args...) where N
    i, j = @index(Global, NTuple)
    @unroll for n in 1:length(c)
        _fill_bottom_halo!(i, j, grid, c[n], bottom_bc[n], args...)
           _fill_top_halo!(i, j, grid, c[n], top_bc[n],    args...)
    end
end

fill_west_and_east_halo!(c, west_bc, east_bc, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, :yz, _fill_west_and_east_halo!, c, west_bc, east_bc, grid, args...; dependencies=dep, kwargs...)

fill_south_and_north_halo!(c, south_bc, north_bc, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, :xz, _fill_south_and_north_halo!, south_bc, north_bc, c, grid, args...; dependencies=dep, kwargs...)

fill_bottom_and_top_halo!(c, bottom_bc, top_bc, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, :xy, _fill_bottom_and_top_halo!, c, bottom_bc, top_bc, grid, args...; dependencies=dep, kwargs...)