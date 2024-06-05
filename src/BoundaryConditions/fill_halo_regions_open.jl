function fill_boundary_normal_velocities!(velocities, clock, fields)
    grid = velocities.u.grid

    fill_open_boundary_regions!(velocities.u, grid, instantiated_location(velocities.u), clock, fields)
    fill_open_boundary_regions!(velocities.v, grid, instantiated_location(velocities.v), clock, fields)
    fill_open_boundary_regions!(velocities.w, grid, instantiated_location(velocities.w), clock, fields)

    return nothing
end

const BoundedGrid = Union{AbstractGrid{<:Any, <:Bounded}, 
                          AbstractGrid{<:Any, <:Any, <:Bounded}, 
                          AbstractGrid{<:Any, <:Any, <:Any, <:Bounded}}

@inline fill_open_boundary_regions!(field, grid, args...) = nothing

@inline fill_open_boundary_regions!(field) = fill_open_boundary_regions!(field, field.grid, instantiated_location(field))

function fill_open_boundary_regions!(field, grid::BoundedGrid, loc, args...)
    arch = architecture(grid)

    left_bc = left_boundary_condition(field, loc)
    right_bc = right_boundary_condition(field, loc)

    launch!(arch, grid, side_plan(loc), fill_open_halo!(loc), field, left_bc, right_bc, loc, grid, args)

    return nothing
end

@inline left_boundary_condition(field, loc::Tuple{Face, Center, Center}) = field.boundary_conditions.west
@inline left_boundary_condition(field, loc::Tuple{Center, Face, Center}) = field.boundary_conditions.south
@inline left_boundary_condition(field, loc::Tuple{Center, Center, Face}) = field.boundary_conditions.bottom

@inline right_boundary_condition(field, loc::Tuple{Face, Center, Center}) = field.boundary_conditions.east
@inline right_boundary_condition(field, loc::Tuple{Center, Face, Center}) = field.boundary_conditions.north
@inline right_boundary_condition(field, loc::Tuple{Center, Center, Face}) = field.boundary_conditions.top

@inline side_plan(loc::Tuple{Face, Center, Center}) = :yz
@inline side_plan(loc::Tuple{Center, Face, Center}) = :xz
@inline side_plan(loc::Tuple{Center, Center, Face}) = :xy

@inline fill_open_halo!(loc::Tuple{Face, Center, Center}) = _fill_west_and_east_open_halo!
@inline fill_open_halo!(loc::Tuple{Center, Face, Center}) = _fill_south_and_north_open_halo!
@inline fill_open_halo!(loc::Tuple{Center, Center, Face}) = _fill_bottom_and_top_open_halo!

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

# fallback for normal boundary conditions

@inline   _fill_west_open_halo!(j, k, grid, c, bc, loc, args...) = nothing
@inline   _fill_east_open_halo!(j, k, grid, c, bc, loc, args...) = nothing
@inline  _fill_south_open_halo!(i, k, grid, c, bc, loc, args...) = nothing
@inline  _fill_north_open_halo!(i, k, grid, c, bc, loc, args...) = nothing
@inline _fill_bottom_open_halo!(i, j, grid, c, bc, loc, args...) = nothing
@inline    _fill_top_open_halo!(i, j, grid, c, bc, loc, args...) = nothing

# and don't do anything on the normal fill call

@inline   _fill_west_halo!(j, k, grid, c, bc, loc, args...) = nothing
@inline   _fill_east_halo!(j, k, grid, c, bc, loc, args...) = nothing
@inline  _fill_south_halo!(i, k, grid, c, bc, loc, args...) = nothing
@inline  _fill_north_halo!(i, k, grid, c, bc, loc, args...) = nothing
@inline _fill_bottom_halo!(i, j, grid, c, bc, loc, args...) = nothing
@inline    _fill_top_halo!(i, j, grid, c, bc, loc, args...) = nothing

# generic for open boundary conditions

@inline   _fill_west_open_halo!(j, k, grid, c, bc::OBC, loc, args...) = @inbounds c[1, j, k]           = getbc(bc, j, k, grid, args...)
@inline   _fill_east_open_halo!(j, k, grid, c, bc::OBC, loc, args...) = @inbounds c[grid.Nx + 1, j, k] = getbc(bc, j, k, grid, args...)
@inline  _fill_south_open_halo!(i, k, grid, c, bc::OBC, loc, args...) = @inbounds c[i, 1, k]           = getbc(bc, i, k, grid, args...)
@inline  _fill_north_open_halo!(i, k, grid, c, bc::OBC, loc, args...) = @inbounds c[i, grid.Ny + 1, k] = getbc(bc, i, k, grid, args...)
@inline _fill_bottom_open_halo!(i, j, grid, c, bc::OBC, loc, args...) = @inbounds c[i, j, 1]           = getbc(bc, i, j, grid, args...)
@inline    _fill_top_open_halo!(i, j, grid, c, bc::OBC, loc, args...) = @inbounds c[i, j, grid.Nz + 1] = getbc(bc, i, j, grid, args...)
