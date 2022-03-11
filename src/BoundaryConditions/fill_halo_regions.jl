using OffsetArrays: OffsetArray
using Oceananigans.Architectures: device_event

#####
##### General halo filling functions
#####

fill_halo_regions!(::Nothing, args...) = nothing

"""
    fill_halo_regions!(fields::Union{Tuple, NamedTuple}, arch, args...)

Fill halo regions for each field in the tuple `fields` according to their boundary
conditions, possibly recursing into `fields` if it is a nested tuple-of-tuples.
"""
function fill_halo_regions!(fields::Union{Tuple, NamedTuple}, arch, args...)
    for field in fields
        fill_halo_regions!(field, arch, args...)
    end
    return nothing
end

# Some fields have `nothing` boundary conditions, such as `FunctionField` and `ZeroField`.
fill_halo_regions!(c::OffsetArray, ::Nothing, args...; kwargs...) = nothing

"Fill halo regions in ``x``, ``y``, and ``z`` for a given field's data."
function fill_halo_regions!(c::OffsetArray, boundary_conditions, arch, grid, args...; kwargs...)

    fill_halos! = [
        fill_west_and_east_halo!,
        fill_south_and_north_halo!,
        fill_bottom_and_top_halo!,
    ]

    boundary_conditions_array_left = [
        boundary_conditions.west,
        boundary_conditions.south,
        boundary_conditions.bottom,
    ]

    boundary_conditions_array_right = [
        boundary_conditions.east,
        boundary_conditions.north,
        boundary_conditions.top,
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

fill_first(bc1::PBC, bc2)      = false
fill_first(bc1, bc2::PBC)      = true
fill_first(bc1::PBC, bc2::PBC) = true
fill_first(bc1, bc2)           = true

#####
##### General fill_halo! kernels
#####

fill_west_and_east_halo!(c, west_bc, east_bc, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, :yz, _fill_west_and_east_halo!, c, grid; dependencies=dep, kwargs...)

fill_south_and_north_halo!(c, south_bc, north_bc, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, :xz, _fill_south_and_north_halo!, c, grid, args...; dependencies=dep, kwargs...)

fill_bottom_and_top_halo!(c, bottom_bc, top_bc, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, :xy, _fill_bottom_and_top_halo!, c, grid, args...; dependencies=dep, kwargs...)

@kernel function _fill_west_and_east_halo!(c, west_bc, east_bc, grid, args...)
    j, k = @index(Global, NTuple)
    _fill_west_halo!(j, k, grid, c, west_bc, args...)
    _fill_east_halo!(j, k, grid, c, east_bc, args...)
end

@kernel function _fill_south_and_north_halo!(c, south_bc, north_bc, grid, args...)
    i, k = @index(Global, NTuple)
    _fill_south_halo!(i, k, grid, c, south_bc, args...)
    _fill_north_halo!(i, k, grid, c, north_bc, args...)
end

@kernel function _fill_bottom_and_top_halo!(c, bottom_bc, top_bc, grid, args...)
    i, j = @index(Global, NTuple)
    _fill_bottom_halo!(c, i, j, grid, bottom_bc, args...)
    _fill_top_halo!(c, i, j, grid, top_bc, args...)
end

#####
##### Tuple fill_halo! kernels
#####

@kernel function _fill_west_and_east_halo!(c::NTuple{N}, west_bc::NTuple{N}, east_bc::NTuple{N}, grid, args...) where N
    j, k = @index(Global, NTuple)
    @unroll for n in 1:N
        _fill_west_halo!(j, k, grid, c[n], west_bc[n], args...)
        _fill_east_halo!(j, k, grid, c[n], east_bc[n], args...)
    end
end

@kernel function _fill_south_and_north_halo!(c::NTuple{N}, south_bc::NTuple{N}, north_bc::NTuple{N}, grid, args...) where N
    i, k = @index(Global, NTuple)
    @unroll for n in 1:N
        _fill_south_halo!(i, k, grid, c[n], south_bc[n], args...)
        _fill_north_halo!(i, k, grid, c[n], north_bc[n], args...)
    end
end

@kernel function _fill_bottom_and_top_halo!(c::NTuple{N}, bottom_bc::NTuple{N}, top_bc::NTuple{N}, grid, args...) where N
    i, j = @index(Global, NTuple)
    @unroll for n in 1:N
        _fill_bottom_halo!(i, j, grid, c[n], bottom_bc[n], args...)
           _fill_top_halo!(i, j, grid, c[n], top_bc[n], args...)
    end
end