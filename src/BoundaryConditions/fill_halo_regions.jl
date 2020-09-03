#####
##### General halo filling functions
#####

fill_halo_regions!(::Nothing, args...) = []

"""
    fill_halo_regions!(fields, arch)

Fill halo regions for each field in the tuple `fields` according to their boundary
conditions, possibly recursing into `fields` if it is a nested tuple-of-tuples.
"""
function fill_halo_regions!(fields::Union{Tuple, NamedTuple}, arch, args...)

    for field in fields
        fill_halo_regions!(field, arch, args...)
    end

    return nothing
end

fill_halo_regions!(field, arch, args...) =
    fill_halo_regions!(field.data, field.boundary_conditions, arch, field.grid, args...)

"Fill halo regions in x, y, and z for a given field."
function fill_halo_regions!(c::AbstractArray, fieldbcs, arch, grid, args...)

    barrier = Event(device(arch))

      west_event =   fill_west_halo!(c, fieldbcs.x.left,   arch, barrier, grid, args...)
      east_event =   fill_east_halo!(c, fieldbcs.x.right,  arch, barrier, grid, args...)
     south_event =  fill_south_halo!(c, fieldbcs.y.left,   arch, barrier, grid, args...)
     north_event =  fill_north_halo!(c, fieldbcs.y.right,  arch, barrier, grid, args...)
    bottom_event = fill_bottom_halo!(c, fieldbcs.z.bottom, arch, barrier, grid, args...)
       top_event =    fill_top_halo!(c, fieldbcs.z.top,    arch, barrier, grid, args...)

    # Wait at the end
    events = [west_event, east_event, south_event, north_event, bottom_event, top_event]
    events = filter(e -> typeof(e) <: Event, events)
    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end
