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

"Fill halo regions in x, y, and z for a given field's data."
function fill_halo_regions!(c::OffsetArray, field_bcs, arch, grid, args...; kwargs...)

    fill_halos! = [
        fill_west_and_east_halo!,
        fill_south_and_north_halo!,
        fill_bottom_and_top_halo!,
    ]

    field_bcs_array_left = [
        field_bcs.west,
        field_bcs.south,
        field_bcs.bottom,
    ]

    field_bcs_array_right = [
        field_bcs.east,
        field_bcs.north,
        field_bcs.top,
    ]

    perm = sortperm(field_bcs_array_left, lt=fill_first)
    fill_halos! = fill_halos![perm]
    bcs_left  = field_bcs_array_left[perm]
    bcs_right = field_bcs_array_right[perm]
    
    event0 = Event(device(arch))
    fill_halo! = fill_halos![1]
    event1     = fill_halo!(c, bcs_left[1], bcs_right[1], arch, event0, grid, args...; kwargs...)   
    wait(device(arch), event1)
    fill_halo! = fill_halos![2]
    event2     = fill_halo!(c, bcs_left[2], bcs_right[2], arch, event1, grid, args...; kwargs...)   
    wait(device(arch), event2)
    fill_halo! = fill_halos![3]
    event3     = fill_halo!(c, bcs_left[3], bcs_right[3], arch, event2, grid, args...; kwargs...)   
    wait(device(arch), event3)

    return nothing
end

# Hacky way to get rid of "Nothing" events
@inline validate_event(event)        = NoneEvent()
@inline validate_event(event::Event) = event

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
##### Halo-filling for nothing boundary conditions
#####

  fill_west_halo!(c, ::Nothing, args...; kwargs...) = NoneEvent()
  fill_east_halo!(c, ::Nothing, args...; kwargs...) = NoneEvent()
 fill_south_halo!(c, ::Nothing, args...; kwargs...) = NoneEvent()
 fill_north_halo!(c, ::Nothing, args...; kwargs...) = NoneEvent()
   fill_top_halo!(c, ::Nothing, args...; kwargs...) = NoneEvent()
fill_bottom_halo!(c, ::Nothing, args...; kwargs...) = NoneEvent()

#####
##### Halo filling order
#####

fill_first(bc1::PBC, bc2)      = false
fill_first(bc1, bc2::PBC)      = true
fill_first(bc1::PBC, bc2::PBC) = true
fill_first(bc1, bc2)           = true
