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
fill_halo_regions!(c::OffsetArray, ::Nothing, args...; kwargs...) = nothing

"Fill halo regions in x, y, and z for a given field's data."
function fill_halo_regions!(c::OffsetArray, field_bcs, arch, grid, args...; kwargs...)

    fill_halos! = [
        fill_bottom_halo!,
        fill_top_halo!,
        fill_south_halo!,
        fill_north_halo!,
        fill_west_halo!,
        fill_east_halo!,
    ]

    field_bcs_array = [
        field_bcs.bottom,
        field_bcs.top,
        field_bcs.south,
        field_bcs.north,
        field_bcs.west,
        field_bcs.east,
    ]

    perm = sortperm(field_bcs_array, lt=fill_first)

    fill_halos! = fill_halos![perm]
    field_bcs_array = field_bcs_array[perm]

    barrier = Event(device(arch))

    # Fill sequentially
    for task_idx = 1:6
        fill_halo! = fill_halos![task_idx]
        bc = field_bcs_array[task_idx]
        barrier = fill_halo!(c, bc, arch, barrier, grid, args...; dependencies=barrier, kwargs...)
    end

    wait(device(arch), barrier)

    return nothing
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

fill_first(bc1::PBC, bc2) = false
fill_first(bc1, bc2::PBC) = true
fill_first(bc1::PBC, bc2::PBC) = true
fill_first(bc1, bc2) = true

