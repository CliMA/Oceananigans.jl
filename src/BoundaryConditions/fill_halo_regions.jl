#####
##### General halo filling functions
#####

fill_halo_regions!(::Nothing, args...) = nothing

"""
    fill_halo_regions!(fields, arch)

Fill halo regions for each field in the tuple `fields` according to their boundary
conditions, possibly recursing into `fields` if it is a nested tuple-of-tuples.
"""
function fill_halo_regions!(fields::NamedTuple, arch, args...)
    for field in fields
        fill_halo_regions!(field, arch, args...)
    end
    return nothing
end

fill_halo_regions!(field, arch, args...) =
    fill_halo_regions!(field.data, field.boundary_conditions, arch, field.grid, args...)

"Fill halo regions in x, y, and z for a given field."
function fill_halo_regions!(c::AbstractArray, fieldbcs, arch, grid, args...)
      fill_west_halo!(c, fieldbcs.x.left,   arch, grid, args...)
      fill_east_halo!(c, fieldbcs.x.right,  arch, grid, args...)
     fill_south_halo!(c, fieldbcs.y.left,   arch, grid, args...)
     fill_north_halo!(c, fieldbcs.y.right,  arch, grid, args...)
    fill_bottom_halo!(c, fieldbcs.z.bottom, arch, grid, args...)
       fill_top_halo!(c, fieldbcs.z.top,    arch, grid, args...)
    return nothing
end
