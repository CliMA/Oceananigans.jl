@inline fill_open_boundary_regions!(field, args...) = 
    fill_open_boundary_regions!(field, field.boundary_conditions, field.indices, instantiated_location(field), field.grid)

"""
    fill_open_boundary_regions!(fields, boundary_conditions, indices, loc, grid, args...; kwargs...)

Fill open boundary halo regions by filling boundary conditions on field faces with `open_fill`. 
"""
function fill_open_boundary_regions!(field, boundary_conditions, indices, loc, grid, args...; kwargs...)
    arch = architecture(grid)

    left_bc = left_velocity_open_boundary_condition(boundary_conditions, loc)
    right_bc = right_velocity_open_boundary_condition(boundary_conditions, loc)

    # gets `open_fill`, the function which fills open boundaries at `loc`, as well as `regular_fill`
    # which is the function which fills non-open boundaries at `loc` which informs `fill_halo_size` 
    open_fill, regular_fill = get_open_halo_filling_functions(loc) 
    fill_size = fill_halo_size(field, regular_fill, indices, boundary_conditions, loc, grid)

    launch!(arch, grid, fill_size, open_fill, field, left_bc, right_bc, loc, grid, args)

    return nothing
end

@inline get_open_halo_filling_functions(loc) = nothing
@inline get_open_halo_filling_functions(::Tuple{Face, Center, Center}) = fill_west_and_east_halo!
@inline get_open_halo_filling_functions(::Tuple{Center, Face, Center}) = fill_south_and_north_halo!
@inline get_open_halo_filling_functions(::Tuple{Center, Center, Face}) = fill_bottom_and_top_halo!

function fill_open_boundary_regions!(fields::Tuple, boundary_conditions, indices, loc, grid, args...; kwargs...) 
    for n in eachindex(fields)
        fill_open_boundary_regions!(fields[n], boundary_conditions[n], indices, loc[n], grid, args...; kwargs...)
    end
    return nothing
end

@inline retrieve_open_bc(bc::OBC) = bc
@inline retrieve_open_bc(bc) = nothing

@inline retrieve_bc(bc::OBC) = nothing

# for regular halo fills, return nothing if the BC is not an OBC
@inline left_open_boundary_condition(boundary_condition, loc) = nothing
@inline left_open_boundary_condition(boundary_conditions, ::Tuple{Face, Center, Center}) = retrieve_open_bc(boundary_conditions.west)
@inline left_open_boundary_condition(boundary_conditions, ::Tuple{Center, Face, Center}) = retrieve_open_bc(boundary_conditions.south)
@inline left_open_boundary_condition(boundary_conditions, ::Tuple{Center, Center, Face}) = retrieve_open_bc(boundary_conditions.bottom)

@inline right_open_boundary_condition(boundary_conditions, loc) = nothing
@inline right_open_boundary_condition(boundary_conditions, ::Tuple{Face, Center, Center}) = retrieve_open_bc(boundary_conditions.east)
@inline right_open_boundary_condition(boundary_conditions, ::Tuple{Center, Face, Center}) = retrieve_open_bc(boundary_conditions.north)
@inline right_open_boundary_condition(boundary_conditions, ::Tuple{Center, Center, Face}) = retrieve_open_bc(boundary_conditions.top)

# Open boundary fill 
@inline   _fill_west_halo!(j, k, grid, c, bc::OBC, loc, args...) = @inbounds c[1, j, k]           = getbc(bc, j, k, grid, args...)
@inline   _fill_east_halo!(j, k, grid, c, bc::OBC, loc, args...) = @inbounds c[grid.Nx + 1, j, k] = getbc(bc, j, k, grid, args...)
@inline  _fill_south_halo!(i, k, grid, c, bc::OBC, loc, args...) = @inbounds c[i, 1, k]           = getbc(bc, i, k, grid, args...)
@inline  _fill_north_halo!(i, k, grid, c, bc::OBC, loc, args...) = @inbounds c[i, grid.Ny + 1, k] = getbc(bc, i, k, grid, args...)
@inline _fill_bottom_halo!(i, j, grid, c, bc::OBC, loc, args...) = @inbounds c[i, j, 1]           = getbc(bc, i, j, grid, args...)
@inline    _fill_top_halo!(i, j, grid, c, bc::OBC, loc, args...) = @inbounds c[i, j, grid.Nz + 1] = getbc(bc, i, j, grid, args...)
