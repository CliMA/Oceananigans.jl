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
@inline validate_event(event) = event

# Fallbacks split into two calls
function fill_west_and_east_halo!(c, west_bc, east_bc, args...; kwargs...)
     west_event = validate_event(fill_west_halo!(c, west_bc, args...; kwargs...))
     east_event = validate_event(fill_east_halo!(c, east_bc, args...; kwargs...))
    multi_event = MultiEvent((west_event, east_event))
    return multi_event
end

function fill_south_and_north_halo!(c, south_bc, north_bc, args...; kwargs...)
    south_event = validate_event(fill_south_halo!(c, south_bc, args...; kwargs...))
    north_event = validate_event(fill_north_halo!(c, north_bc, args...; kwargs...))
    multi_event = MultiEvent((south_event, north_event))
    return multi_event
end

function fill_bottom_and_top_halo!(c, bottom_bc, top_bc, args...; kwargs...)
    bottom_event = validate_event(fill_bottom_halo!(c, bottom_bc, args...; kwargs...))
       top_event = validate_event(fill_top_halo!(c, top_bc, args...; kwargs...))
     multi_event = MultiEvent((bottom_event, top_event))
     return multi_event
end

#####
##### Halo filling order
#####

fill_first(bc1::PBC, bc2)      = false
fill_first(bc1, bc2::PBC)      = true
fill_first(bc1::PBC, bc2::PBC) = true
fill_first(bc1, bc2)           = true
