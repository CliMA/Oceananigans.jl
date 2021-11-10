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
    field_bcs_array_left  = field_bcs_array_left[perm]
    field_bcs_array_right = field_bcs_array_right[perm]
   
    events = device_event(arch)
    for task = 1:3

        barrier = device_event(arch)

        fill_halo!  = fill_halos![task]
        bc_left     = field_bcs_array_left[task]
        bc_right    = field_bcs_array_right[task]

        events      = fill_halo!(c, bc_left, bc_right, arch, barrier, grid, args...; kwargs...)
       
        # Three different ways to synchronize the streams associated with the boundary 
        # Some work, some will not work:
        
        # it most likely has to do with the fact that the events live in 
        # scope: 
        
        # Most likely (again), kernel is launching kernels on different streams, and,
        # as such the only way to synchronize is to synchronize on the host (slower but correct) 
        # 
        # If we try to force the kernels in the same stream or try to keep all the 
        
        # if events != NoneEvent() 
        #     if hasproperty(events, :events) 
        #         wait(device(arch), events)
        #      else
        #         arch isa CPU ? wait(events) : CUDA.synchronize(events.event)
        #     end
        # end
        wait(device(arch), events)
    end



    return nothing
end

# Fallbacks split into two calls
function fill_west_and_east_halo!(c, west_bc, east_bc, arch, args...; kwargs...)
     west_event = fill_west_halo!(c, west_bc, args...; kwargs...)
     east_event = fill_east_halo!(c, east_bc, args...; kwargs...)
    multi_event = MultiEvent((west_event, east_event))
    return multi_event
end

function fill_south_and_north_halo!(c, south_bc, north_bc, args...; kwargs...)
    south_event = fill_south_halo!(c, south_bc, args...; kwargs...)
    north_event = fill_north_halo!(c, north_bc, args...; kwargs...)
    multi_event = MultiEvent((south_event, north_event))
    return multi_event
end

function fill_bottom_and_top_halo!(c, bottom_bc, top_bc, args...; kwargs...)
    bottom_event = fill_bottom_halo!(c, bottom_bc, args...; kwargs...)
       top_event = fill_top_halo!(c, top_bc, args...; kwargs...)
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
