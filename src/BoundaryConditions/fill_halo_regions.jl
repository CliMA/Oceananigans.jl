#####
##### General halo filling functions
#####

fill_halo_regions!(::Nothing, args...) = nothing

"""
    fill_halo_regions!(fields, bcs, arch, grid)

Fill halo regions for each field in the tuple `fields` according
to the single instance of `FieldBoundaryConditions` in `bcs`, possibly recursing into
`fields` if it is a nested tuple-of-tuples.
"""
function fill_halo_regions!(fields::NamedTuple, bcs::FieldBoundaryConditions, arch, grid, args...)
    for field in fields
        fill_halo_regions!(field, bcs, arch, grid, args...)
    end
    return nothing
end

"Fill halo regions in x, y, and z for a given field."
function fill_halo_regions!(c::AbstractArray, fieldbcs, arch, grid, args...)
      fill_west_halo!(c, fieldbcs.x.left,  arch, grid, args...)
      fill_east_halo!(c, fieldbcs.x.right, arch, grid, args...)

     fill_south_halo!(c, fieldbcs.y.left,  arch, grid, args...)
     fill_north_halo!(c, fieldbcs.y.right, arch, grid, args...)

     fill_bottom_halo!(c, fieldbcs.z.bottom, arch, grid, args...)
        fill_top_halo!(c, fieldbcs.z.top,    arch, grid, args...)
    return nothing
end

#####
##### Convenience methods for filling halo regions of tupled fields
#####

function tupled_fill_halo_regions!(fields, bcs, arch, grid, args...)
    for (field, fieldbcs) in zip(fields, bcs)
        fill_halo_regions!(field, fieldbcs, arch, grid, args...)
    end
    return nothing
end

"""
    fill_halo_regions!(fields, bcs, arch, grid)

Fill halo regions for all fields in the `Tuple` `fields` according
to the corresponding `Tuple` of `bcs`.
"""
fill_halo_regions!(fields::Tuple, bcs::Tuple, arch, grid, args...) =
    tupled_fill_halo_regions!(fields, bcs, arch, grid, args...)

"""
    fill_halo_regions!(fields, bcs, arch, grid)

Fill halo regions for all fields in the `NamedTuple` `fields` according
to the corresponding `NamedTuple` of `bcs`.
"""
fill_halo_regions!(fields::NamedTuple{S}, bcs::NamedTuple{S}, arch, grid, args...) where S =
    tupled_fill_halo_regions!(fields, bcs, arch, grid, args...)
