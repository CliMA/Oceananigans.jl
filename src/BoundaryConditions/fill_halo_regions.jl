#=
using OffsetArrays: OffsetArray

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
function fill_halo_regions!(c::OffsetArray, fieldbcs, arch, grid, args...; kwargs...)

    barrier = Event(device(arch))

      west_event =   fill_west_halo!(c, fieldbcs.west,   arch, barrier, grid, args...; kwargs...)
      east_event =   fill_east_halo!(c, fieldbcs.east,   arch, barrier, grid, args...; kwargs...)
     south_event =  fill_south_halo!(c, fieldbcs.south,  arch, barrier, grid, args...; kwargs...)
     north_event =  fill_north_halo!(c, fieldbcs.north,  arch, barrier, grid, args...; kwargs...)
    bottom_event = fill_bottom_halo!(c, fieldbcs.bottom, arch, barrier, grid, args...; kwargs...)
       top_event =    fill_top_halo!(c, fieldbcs.top,    arch, barrier, grid, args...; kwargs...)

    # Wait at the end
    events = [west_event, east_event, south_event, north_event, bottom_event, top_event]
    events = filter(e -> e isa Event, events)
    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Halo-filling for nothing boundary conditions
#####

  fill_west_halo!(c, ::Nothing, args...; kwargs...) = nothing
  fill_east_halo!(c, ::Nothing, args...; kwargs...) = nothing
 fill_south_halo!(c, ::Nothing, args...; kwargs...) = nothing
 fill_north_halo!(c, ::Nothing, args...; kwargs...) = nothing
   fill_top_halo!(c, ::Nothing, args...; kwargs...) = nothing
fill_bottom_halo!(c, ::Nothing, args...; kwargs...) = nothing
=#

using Oceananigans.Architectures: device_event
using OffsetArrays: OffsetArray

#####
##### General halo filling functions
#####

fill_halo_regions!(::Nothing, args...) = []

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
fill_halo_regions!(c::OffsetArray, ::Nothing, args...; kwargs...) = NoneEvent()

"Fill halo regions in x, y, and z for a given field's data."
function fill_halo_regions!(c::OffsetArray, bcs, arch, grid, args...; kwargs...)

    barrier = device_event(arch)

    bt_events = fill_bottom_and_top_halo!(c,  bcs.bottom, bcs.top,   arch, device_event(arch), grid, args...; kwargs...)
    sn_events = fill_south_and_north_halo!(c, bcs.south,  bcs.north, arch, south_north_events, grid, args...; kwargs...)
    we_events = fill_west_and_east_halo!(c,   bcs.west,   bcs.east,  arch, west_east_events, grid, args...; kwargs...)

    # Wait at the end
    events = (west_event, east_event, south_event, north_event, bottom_event, top_event)

    wait(device(arch), MultiEvent(events))

    return NoneEvent()
end

#####
##### Default halo filling launches two kernels
#####

# Fallbacks split into two calls
function fill_west_and_east_halo!(c, west_bc, east_bc, args...; kwargs...)
    west_event = fill_west_halo!(c, west_bc, args...; kwargs...)
    east_event = fill_east_halo!(c, east_bc, args...; kwargs...)
    return west_event, east_event
end

function fill_south_and_north_halo!(c, south_bc, north_bc, args...; kwargs...)
    south_event = fill_south_halo!(c, south_bc, args...; kwargs...)
    north_event = fill_north_halo!(c, north_bc, args...; kwargs...)
    return south_event, north_event
end

function fill_bottom_and_top_halo!(c, bottom_bc, top_bc, args...; kwargs...)
    bottom_event = fill_bottom_halo!(c, bottom_bc, args...; kwargs...)
    top_event = fill_top_halo!(c, top_bc, args...; kwargs...)
    return bottom_event, top_event
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
