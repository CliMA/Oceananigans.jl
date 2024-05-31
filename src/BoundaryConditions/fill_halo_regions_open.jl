function fill_bounded_wall_normal_halo_regions!(velocities, clock, fields)
    grid = velocities.u.grid

    TX, TY, TZ = topology(grid)

    if TX() isa Bounded
        fill_u_open_halo_regions!(velocities.u, grid, clock, fields)
    end

    if TY() isa Bounded
        fill_v_open_halo_regions!(velocities.v, grid, clock, fields)
    end

    if TZ() isa Bounded
        fill_w_open_halo_regions!(velocities.w, grid, clock, fields)
    end

    return nothing
end

function fill_u_open_halo_regions!(u, grid, args...)
    arch = architecture(grid)
    
    west_bc = u.boundary_conditions.west
    east_bc = u.boundary_conditions.east

    launch!(arch, grid, :yz, _fill_west_and_east_open_halo!, u, west_bc, east_bc, location(u), grid, args)
end

function fill_v_open_halo_regions!(v, grid, args...)
    arch = architecture(grid)

    south_bc = v.boundary_conditions.south
    north_bc = v.boundary_conditions.north

    launch!(arch, grid, :yz, _fill_south_and_north_open_halo!, v, south_bc, north_bc, location(v), grid, args)
end

function fill_w_open_halo_regions!(w, grid, args...)
    arch = architecture(grid)

    bottom_bc = w.boundary_conditions.bottom
    top_bc = w.boundary_conditions.top

    launch!(arch, grid, :yz, _fill_bottom_and_top_open_halo!, w, bottom_bc, top_bc, location(w), grid, args)
end

@kernel function _fill_west_and_east_open_halo!(c, west_bc, east_bc, loc, grid, args) 
    j, k = @index(Global, NTuple)
    _fill_west_open_halo!(j, k, grid, c, west_bc, loc, args...)
    _fill_east_open_halo!(j, k, grid, c, east_bc, loc, args...)
end

@kernel function _fill_south_and_top_open_halo!(c, south_bc, top_bc, loc, grid, args)
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
