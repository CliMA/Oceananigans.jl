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


fill_open_boundary_regions!(fields::NTuple, boundary_conditions, indices, loc, grid, args...; kwargs...) = 
    ntuple(n->fill_open_boundary_regions!(fields[n], boundary_conditions[n], indices, loc[n], grid, args...; kwargs...), Val(length(fields)))

# for regular halo fills
@inline left_velocity_open_boundary_condition(boundary_condition, loc) = nothing
@inline left_velocity_open_boundary_condition(boundary_conditions, ::Tuple{Face, Center, Center}) = boundary_conditions.west
@inline left_velocity_open_boundary_condition(boundary_conditions, ::Tuple{Center, Face, Center}) = boundary_conditions.south
@inline left_velocity_open_boundary_condition(boundary_conditions, ::Tuple{Center, Center, Face}) = boundary_conditions.bottom

@inline right_velocity_open_boundary_condition(boundary_conditions, loc) = nothing
@inline right_velocity_open_boundary_condition(boundary_conditions, ::Tuple{Face, Center, Center}) = boundary_conditions.east
@inline right_velocity_open_boundary_condition(boundary_conditions, ::Tuple{Center, Face, Center}) = boundary_conditions.north
@inline right_velocity_open_boundary_condition(boundary_conditions, ::Tuple{Center, Center, Face}) = boundary_conditions.top

# for multi region halo fills
@inline left_velocity_open_boundary_condition(boundary_conditions::Tuple, ::Tuple{Face, Center, Center}) = @inbounds boundary_conditions[1]
@inline left_velocity_open_boundary_condition(boundary_conditions::Tuple, ::Tuple{Center, Face, Center}) = @inbounds boundary_conditions[1]
@inline left_velocity_open_boundary_condition(boundary_conditions::Tuple, ::Tuple{Center, Center, Face}) = @inbounds boundary_conditions[1]

@inline right_velocity_open_boundary_condition(boundary_conditions::Tuple, ::Tuple{Face, Center, Center}) = @inbounds boundary_conditions[2]
@inline right_velocity_open_boundary_condition(boundary_conditions::Tuple, ::Tuple{Center, Face, Center}) = @inbounds boundary_conditions[2]
@inline right_velocity_open_boundary_condition(boundary_conditions::Tuple, ::Tuple{Center, Center, Face}) = @inbounds boundary_conditions[2]

@inline get_open_halo_filling_functions(loc) = _no_fill!, _no_fill!
@inline get_open_halo_filling_functions(::Tuple{Face, Center, Center}) = _fill_west_and_east_open_halo!, fill_west_and_east_halo!
@inline get_open_halo_filling_functions(::Tuple{Center, Face, Center}) = _fill_south_and_north_open_halo!, fill_south_and_north_halo!
@inline get_open_halo_filling_functions(::Tuple{Center, Center, Face}) = _fill_bottom_and_top_open_halo!, fill_bottom_and_top_halo!

@kernel _no_fill!(args...) = nothing

@inline fill_halo_size(field, ::typeof(_no_fill!), args...) = (0, 0)

@kernel function _fill_west_and_east_open_halo!(c, west_bc, east_bc, loc, grid, args) 
    j, k = @index(Global, NTuple)
    _fill_west_open_halo!(j, k, grid, c, west_bc, loc, args...)
    _fill_east_open_halo!(j, k, grid, c, east_bc, loc, args...)
end

@kernel function _fill_south_and_north_open_halo!(c, south_bc, north_bc, loc, grid, args)
    i, k = @index(Global, NTuple)
    _fill_south_open_halo!(i, k, grid, c, south_bc, loc, args...)
    _fill_north_open_halo!(i, k, grid, c, north_bc, loc, args...)
end

@kernel function _fill_bottom_and_top_open_halo!(c, bottom_bc, top_bc, loc, grid, args)
    i, j = @index(Global, NTuple)
    _fill_bottom_open_halo!(i, j, grid, c, bottom_bc, loc, args...)
       _fill_top_open_halo!(i, j, grid, c, top_bc,    loc, args...)
end

# Generic fallback

@inline   _fill_west_open_halo!(j, k, grid, c, bc, loc, args...) = nothing
@inline   _fill_east_open_halo!(j, k, grid, c, bc, loc, args...) = nothing
@inline  _fill_south_open_halo!(i, k, grid, c, bc, loc, args...) = nothing
@inline  _fill_north_open_halo!(i, k, grid, c, bc, loc, args...) = nothing
@inline _fill_bottom_open_halo!(i, j, grid, c, bc, loc, args...) = nothing
@inline    _fill_top_open_halo!(i, j, grid, c, bc, loc, args...) = nothing

# Open boundary condition fallback

@inline   _fill_west_open_halo!(j, k, grid, c, bc::OBC, loc, args...) = @inbounds c[1, j, k]           = getbc(bc, j, k, grid, args...)
@inline   _fill_east_open_halo!(j, k, grid, c, bc::OBC, loc, args...) = @inbounds c[grid.Nx + 1, j, k] = getbc(bc, j, k, grid, args...)
@inline  _fill_south_open_halo!(i, k, grid, c, bc::OBC, loc, args...) = @inbounds c[i, 1, k]           = getbc(bc, i, k, grid, args...)
@inline  _fill_north_open_halo!(i, k, grid, c, bc::OBC, loc, args...) = @inbounds c[i, grid.Ny + 1, k] = getbc(bc, i, k, grid, args...)
@inline _fill_bottom_open_halo!(i, j, grid, c, bc::OBC, loc, args...) = @inbounds c[i, j, 1]           = getbc(bc, i, j, grid, args...)
@inline    _fill_top_open_halo!(i, j, grid, c, bc::OBC, loc, args...) = @inbounds c[i, j, grid.Nz + 1] = getbc(bc, i, j, grid, args...)

# Regular boundary fill defaults

@inline   _fill_west_halo!(j, k, grid, c, bc::OBC, loc, args...) = _fill_west_open_halo!(j, k, grid, c, bc, loc, args...)
@inline   _fill_east_halo!(j, k, grid, c, bc::OBC, loc, args...) = _fill_east_open_halo!(j, k, grid, c, bc, loc, args...)
@inline  _fill_south_halo!(i, k, grid, c, bc::OBC, loc, args...) = _fill_south_open_halo!(i, k, grid, c, bc, loc, args...)
@inline  _fill_north_halo!(i, k, grid, c, bc::OBC, loc, args...) = _fill_north_open_halo!(i, k, grid, c, bc, loc, args...)
@inline _fill_bottom_halo!(i, j, grid, c, bc::OBC, loc, args...) = _fill_bottom_open_halo!(i, j, grid, c, bc, loc, args...)
@inline    _fill_top_halo!(i, j, grid, c, bc::OBC, loc, args...) = _fill_top_open_halo!(i, j, grid, c, bc, loc, args...)

# Regular boundary fill for wall normal velocities

@inline   _fill_west_halo!(j, k, grid, c, bc::OBC, ::Tuple{Face, <:Any, <:Any}, args...) = nothing
@inline   _fill_east_halo!(j, k, grid, c, bc::OBC, ::Tuple{Face, <:Any, <:Any}, args...) = nothing
@inline  _fill_south_halo!(i, k, grid, c, bc::OBC, ::Tuple{<:Any, Face, <:Any}, args...) = nothing
@inline  _fill_north_halo!(i, k, grid, c, bc::OBC, ::Tuple{<:Any, Face, <:Any}, args...) = nothing
@inline _fill_bottom_halo!(i, j, grid, c, bc::OBC, ::Tuple{<:Any, <:Any, Face}, args...) = nothing
@inline    _fill_top_halo!(i, j, grid, c, bc::OBC, ::Tuple{<:Any, <:Any, Face}, args...) = nothing
