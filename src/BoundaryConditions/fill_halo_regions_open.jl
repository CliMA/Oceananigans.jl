const BoundedGrid = Union{AbstractGrid{<:Any, <:Bounded}, 
                          AbstractGrid{<:Any, <:Any, <:Bounded}, 
                          AbstractGrid{<:Any, <:Any, <:Any, <:Bounded}}

@inline fill_open_boundary_regions!(field, args...) = 
    fill_open_boundary_regions!(field, field.boundary_conditions, field.indices, instantiated_location(field), field.grid)

# what is this for
#@inline fill_open_boundary_regions!(field, grid::BoundedGrid, loc, args...) = fill_open_boundary_regions!(field, field.boundary_conditions, field.indices, loc, grid, args...)

function fill_open_boundary_regions!(field, boundary_conditions, indices, loc, grid, args...; kwargs...)
    arch = architecture(grid)

    left_bc = left_boundary_condition(boundary_conditions, loc)
    right_bc = right_boundary_condition(boundary_conditions, loc)

    open_fill, normal_fill = fill_open_halo(loc) 
    fill_size = fill_halo_size(field, normal_fill, indices, boundary_conditions, loc, grid)

    launch!(arch, grid, fill_size, open_fill, field, left_bc, right_bc, loc, grid, args)

    return nothing
end

fill_open_boundary_regions!(fields::NTuple, boundary_conditions, indices, loc, grid, args...; kwargs...) =
    [fill_open_boundary_regions!(field, boundary_conditions[n], indices, loc[n], grid, args...; kwargs...) for (n, field) in enumerate(fields)]

@inline left_boundary_condition(boundary_conditions, loc) = nothing
@inline left_boundary_condition(boundary_conditions, loc::Tuple{Face, Center, Center}) = boundary_conditions.west
@inline left_boundary_condition(boundary_conditions, loc::Tuple{Center, Face, Center}) = boundary_conditions.south
@inline left_boundary_condition(boundary_conditions, loc::Tuple{Center, Center, Face}) = boundary_conditions.bottom

@inline right_boundary_condition(boundary_conditions, loc) = nothing
@inline right_boundary_condition(boundary_conditions, loc::Tuple{Face, Center, Center}) = boundary_conditions.east
@inline right_boundary_condition(boundary_conditions, loc::Tuple{Center, Face, Center}) = boundary_conditions.north
@inline right_boundary_condition(boundary_conditions, loc::Tuple{Center, Center, Face}) = boundary_conditions.top

@inline fill_open_halo(loc) = _no_fill!, _no_fill!
@inline fill_open_halo(loc::Tuple{Face, Center, Center}) = _fill_west_and_east_open_halo!, fill_west_and_east_halo!
@inline fill_open_halo(loc::Tuple{Center, Face, Center}) = _fill_south_and_north_open_halo!, fill_south_and_north_halo!
@inline fill_open_halo(loc::Tuple{Center, Center, Face}) = _fill_bottom_and_top_open_halo!, fill_bottom_and_top_halo!

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

# fallback for non-open boundary conditions

@inline   _fill_west_open_halo!(j, k, grid, c, bc, loc, args...) = nothing
@inline   _fill_east_open_halo!(j, k, grid, c, bc, loc, args...) = nothing
@inline  _fill_south_open_halo!(i, k, grid, c, bc, loc, args...) = nothing
@inline  _fill_north_open_halo!(i, k, grid, c, bc, loc, args...) = nothing
@inline _fill_bottom_open_halo!(i, j, grid, c, bc, loc, args...) = nothing
@inline    _fill_top_open_halo!(i, j, grid, c, bc, loc, args...) = nothing

# and don't do anything on the non-open boundary fill call

@inline   _fill_west_halo!(j, k, grid, c, bc, loc, args...) = nothing
@inline   _fill_east_halo!(j, k, grid, c, bc, loc, args...) = nothing
@inline  _fill_south_halo!(i, k, grid, c, bc, loc, args...) = nothing
@inline  _fill_north_halo!(i, k, grid, c, bc, loc, args...) = nothing
@inline _fill_bottom_halo!(i, j, grid, c, bc, loc, args...) = nothing
@inline    _fill_top_halo!(i, j, grid, c, bc, loc, args...) = nothing

# except when its a center field (so presumably are tracers)

@inline   _fill_west_halo!(j, k, grid, c, bc::OBC, loc::Tuple{Center, Center, Center}, args...) = @inbounds c[1, j, k]           = getbc(bc, j, k, grid, args...)
@inline   _fill_east_halo!(j, k, grid, c, bc::OBC, loc::Tuple{Center, Center, Center}, args...) = @inbounds c[grid.Nx + 1, j, k] = getbc(bc, j, k, grid, args...)
@inline  _fill_south_halo!(i, k, grid, c, bc::OBC, loc::Tuple{Center, Center, Center}, args...) = @inbounds c[i, 1, k]           = getbc(bc, i, k, grid, args...)
@inline  _fill_north_halo!(i, k, grid, c, bc::OBC, loc::Tuple{Center, Center, Center}, args...) = @inbounds c[i, grid.Ny + 1, k] = getbc(bc, i, k, grid, args...)
@inline _fill_bottom_halo!(i, j, grid, c, bc::OBC, loc::Tuple{Center, Center, Center}, args...) = @inbounds c[i, j, 1]           = getbc(bc, i, j, grid, args...)
@inline    _fill_top_halo!(i, j, grid, c, bc::OBC, loc::Tuple{Center, Center, Center}, args...) = @inbounds c[i, j, grid.Nz + 1] = getbc(bc, i, j, grid, args...)

# generic for open boundary conditions

@inline   _fill_west_open_halo!(j, k, grid, c, bc::OBC, loc, args...) = @inbounds c[1, j, k]           = getbc(bc, j, k, grid, args...)
@inline   _fill_east_open_halo!(j, k, grid, c, bc::OBC, loc, args...) = @inbounds c[grid.Nx + 1, j, k] = getbc(bc, j, k, grid, args...)
@inline  _fill_south_open_halo!(i, k, grid, c, bc::OBC, loc, args...) = @inbounds c[i, 1, k]           = getbc(bc, i, k, grid, args...)
@inline  _fill_north_open_halo!(i, k, grid, c, bc::OBC, loc, args...) = @inbounds c[i, grid.Ny + 1, k] = getbc(bc, i, k, grid, args...)
@inline _fill_bottom_open_halo!(i, j, grid, c, bc::OBC, loc, args...) = @inbounds c[i, j, 1]           = getbc(bc, i, j, grid, args...)
@inline    _fill_top_open_halo!(i, j, grid, c, bc::OBC, loc, args...) = @inbounds c[i, j, grid.Nz + 1] = getbc(bc, i, j, grid, args...)
