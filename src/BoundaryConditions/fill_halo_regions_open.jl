#####
##### Outer functions for filling halo regions for open boundary conditions.
#####

# TODO: support true open boundary conditions.
# For this we need to have separate functions for each of the six boundaries,
# and need to unroll a loop over the boundary normal direction.
# The syntax for `getbc` is also different for OpenBoundaryCondition than for others,
# because the boundary-normal index can vary (and array boundary conditions need to be
# 3D in general).


@inline _fill_open_west_halo!(i, j, k, grid, u) = @inbounds u[1-i, j, k]         = 2 * u[1, j, k]         - u[1+i, j, k]
@inline _fill_open_east_halo!(i, j, k, grid, u) = @inbounds u[grid.Nx+1+i, j, k] = 2 * u[grid.Nx+1, j, k] - u[i+grid.Nz+1-i, j, k]

@inline _fill_open_south_halo!(i, j, k, grid, v) = @inbounds v[i, 1-j, k]         = 2 * v[i, 1, k]         - v[i, 1+j, k]
@inline _fill_open_north_halo!(i, j, k, grid, v) = @inbounds v[i, grid.Ny+1+j, k] = 2 * v[i, grid.Ny+1, k] - v[i, grid.Ny+1-j, k]

@inline _fill_open_bottom_halo!(i, j, k, grid, w) = @inbounds w[i, j, 1-k]         = 2 * w[i, j, 1]         - w[i, j, 1+k]
@inline _fill_open_top_halo!(i, j, k, grid, w)    = @inbounds w[i, j, grid.Nz+1+k] = 2 * w[i, j, grid.Nz+1] - w[i, j, grid.Nz+1-k]

@kernel function set_west_u!(u, i_boundary, bc, grid, args...)
    j, k = @index(Global, NTuple)
    @inbounds u[i_boundary, j, k] = getbc(bc, j, k, grid, args...)

    @unroll for i in 1:grid.Hx
        _fill_open_west_halo!(i, j, k, grid, u)
    end
end

@kernel function set_east_u!(u, i_boundary, bc, grid, args...)
    j, k = @index(Global, NTuple)
    @inbounds u[i_boundary, j, k] = getbc(bc, j, k, grid, args...)

    @unroll for i in 1:grid.Hx
        _fill_open_east_halo!(i, j, k, grid, u)
    end
end

@kernel function set_south_v!(v, j_boundary, bc, grid, args...)
    i, k = @index(Global, NTuple)
    @inbounds v[i, j_boundary, k] = getbc(bc, i, k, grid, args...)

    @unroll for j in 1:grid.Hy
        _fill_open_south_halo!(i, j, k, grid, v)
    end
end

@kernel function set_north_v!(v, j_boundary, bc, grid, args...)
    i, k = @index(Global, NTuple)
    @inbounds v[i, j_boundary, k] = getbc(bc, i, k, grid, args...)
    
    @unroll for j in 1:grid.Hy
        _fill_open_north_halo!(i, j, k, grid, v)
    end
end

@kernel function set_bottom_w!(w, k_boundary, bc, grid, args...)
    i, j = @index(Global, NTuple)
    @inbounds w[i, j, k_boundary] = getbc(bc, i, j, grid, args...)
    
    @unroll for k in 1:grid.Hz
        _fill_open_bottom_halo!(i, j, k, grid, w)
    end
end

@kernel function set_top_w!(w, k_boundary, bc, grid, args...)
    i, j = @index(Global, NTuple)
    @inbounds w[i, j, k_boundary] = getbc(bc, i, j, grid, args...)
    
    @unroll for k in 1:grid.Hz
        _fill_open_top_halo!(i, j, k, grid, w)
    end
end

  @inline fill_west_halo!(u, bc::OBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :yz, set_west_u!,   u,           1, bc, grid, args...; dependencies=dep, kwargs...)
  @inline fill_east_halo!(u, bc::OBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :yz, set_east_u!,   u, grid.Nx + 1, bc, grid, args...; dependencies=dep, kwargs...)
 @inline fill_south_halo!(v, bc::OBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xz, set_south_v!,  v,           1, bc, grid, args...; dependencies=dep, kwargs...)
 @inline fill_north_halo!(v, bc::OBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xz, set_north_v!,  v, grid.Ny + 1, bc, grid, args...; dependencies=dep, kwargs...)
@inline fill_bottom_halo!(w, bc::OBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xy, set_bottom_w!, w,           1, bc, grid, args...; dependencies=dep, kwargs...)
   @inline fill_top_halo!(w, bc::OBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xy, set_top_w!,    w, grid.Nz + 1, bc, grid, args...; dependencies=dep, kwargs...)


@kernel function set_west_and_east_u!(u, west_bc, east_bc, grid, args...)
    j, k = @index(Global, NTuple)

    i_west = 1
    i_east = grid.Nx + 1

    @inbounds begin
        u[i_west, j, k] = getbc(west_bc, j, k, grid, args...)
        u[i_east, j, k] = getbc(east_bc, j, k, grid, args...)
    end

    @unroll for i in 1:grid.Hx
        _fill_open_west_halo!(i, j, k, grid, u)
        _fill_open_east_halo!(i, j, k, grid, u)
    end
end

@kernel function set_south_and_north_v!(v, south_bc, north_bc, grid, args...)
    i, k = @index(Global, NTuple)

    j_south = 1
    j_north = grid.Ny + 1

    @inbounds begin
        v[i, j_south, k] = getbc(south_bc, i, k, grid, args...)
        v[i, j_north, k] = getbc(north_bc, i, k, grid, args...)
    end

    @unroll for j in 1:grid.Hy
        _fill_open_south_halo!(i, j, k, grid, v)
        _fill_open_north_halo!(i, j, k, grid, v)
    end
end

@kernel function set_bottom_and_top_w!(w, bottom_bc, top_bc, grid, args...)
    i, j = @index(Global, NTuple)

    k_bottom = 1
    k_top = grid.Nz + 1

    @inbounds begin
        w[i, j, k_bottom] = getbc(bottom_bc, i, j, grid, args...)
        w[i, j, k_top]    = getbc(top_bc, i, j, grid, args...)
    end    
    
    @unroll for k in 1:grid.Hz
        _fill_open_bottom_halo!(i, j, k, grid, w)
        _fill_open_top_halo!(i, j, k, grid, w)
    end
end

fill_west_and_east_halo!(u, west_bc::OBC, east_bc::OBC, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, :yz, set_west_and_east_u!, u, west_bc, east_bc, grid, args...; dependencies=dep, kwargs...)

fill_south_and_north_halo!(v, south_bc::OBC, north_bc::OBC, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, :xz, set_south_and_north_v!, v, south_bc, north_bc, grid, args...; dependencies=dep, kwargs...)

fill_top_and_bottom_halo!(w, bottom_bc::OBC, top_bc::OBC, arch, dep, grid, args...; kwargs...) =
    launch!(arch, grid, :xy, set_bottom_and_top_w!, w, bottom_bc, top_bc, grid, args...; dependencies=dep, kwargs...)


