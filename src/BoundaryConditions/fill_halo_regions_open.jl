# Open boundary fill — set the boundary value, then extrapolate (zero gradient) into deeper halo cells
@inline function _fill_west_halo!(j, k, grid, c, bc::OBC, loc, args...)
    @inbounds c[1, j, k] = getbc(bc, j, k, grid, args...)
    @inbounds for h in 1:grid.Hx
        c[1 - h, j, k] = c[1, j, k]
    end
end

@inline function _fill_east_halo!(j, k, grid, c, bc::OBC, loc, args...)
    @inbounds c[grid.Nx + 1, j, k] = getbc(bc, j, k, grid, args...)
    @inbounds for h in 2:grid.Hx
        c[grid.Nx + h, j, k] = c[grid.Nx + 1, j, k]
    end
end

@inline function _fill_south_halo!(i, k, grid, c, bc::OBC, loc, args...)
    @inbounds c[i, 1, k] = getbc(bc, i, k, grid, args...)
    @inbounds for h in 1:grid.Hy
        c[i, 1 - h, k] = c[i, 1, k]
    end
end

@inline function _fill_north_halo!(i, k, grid, c, bc::OBC, loc, args...)
    @inbounds c[i, grid.Ny + 1, k] = getbc(bc, i, k, grid, args...)
    @inbounds for h in 2:grid.Hy
        c[i, grid.Ny + h, k] = c[i, grid.Ny + 1, k]
    end
end

@inline function _fill_bottom_halo!(i, j, grid, c, bc::OBC, loc, args...)
    @inbounds c[i, j, 1] = getbc(bc, i, j, grid, args...)
    @inbounds for h in 1:grid.Hz
        c[i, j, 1 - h] = c[i, j, 1]
    end
end

@inline function _fill_top_halo!(i, j, grid, c, bc::OBC, loc, args...)
    @inbounds c[i, j, grid.Nz + 1] = getbc(bc, i, j, grid, args...)
    @inbounds for h in 2:grid.Hz
        c[i, j, grid.Nz + h] = c[i, j, grid.Nz + 1]
    end
end

@inline function fill_halo_event!(c, kernel!, bcs::Tuple{<:OBC, <:OBC}, loc, grid, args...; fill_open_bcs=true, kwargs...)
    if fill_open_bcs
        return kernel!(c, bcs[1], bcs[2], loc, grid, Tuple(args))
    end
    return nothing
end

@inline function fill_halo_event!(c, kernel!, bcs::Tuple{<:OBC}, loc, grid, args...; fill_open_bcs=true, kwargs...)
    if fill_open_bcs
        return kernel!(c, bcs[1], loc, grid, Tuple(args))
    end
    return nothing
end

@inline function fill_halo_event!(c, kernel!, bc::OBC, loc, grid, args...; fill_open_bcs=true, kwargs...)
    if fill_open_bcs
        return kernel!(c, bc, loc, grid, Tuple(args))
    end
    return nothing
end
