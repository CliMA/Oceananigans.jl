#####
##### Outer functions for filling halo regions for open boundary conditions.
#####

# TODO: support true open boundary conditions.
# For this we need to have separate functions for each of the six boundaries,
# and need to unroll a loop over the boundary normal direction.
# The syntax for `getbc` is also different for OpenBoundaryCondition than for others,
# because the boundary-normal index can vary (and array boundary conditions need to be
# 3D in general).

@kernel function set_west_or_east_u!(u, offset, i_boundary, bc, grid, args) 
    j, k = @index(Global, NTuple)
    j′ = j + offset[1]
    k′ = k + offset[2]
@inbounds u[i_boundary, j′, k′] = getbc(bc, j′, k′, grid, args...)
end

@kernel function set_south_or_north_v!(v, offset, j_boundary, bc, grid, args)
    i, k = @index(Global, NTuple)
    i′ = i + offset[1]
    k′ = k + offset[2]
@inbounds v[i′, j_boundary, k′] = getbc(bc, i′, k′, grid, args...)
end

@kernel function set_bottom_or_top_w!(w, offset, k_boundary, bc, grid, args) 
    i, j = @index(Global, NTuple)
    i′ = i + offset[1]
    j′ = j + offset[2]
@inbounds w[i′, j′, k_boundary] = getbc(bc, i′, j′, grid, args...)
end

@inline   fill_west_halo!(u, bc::OBC, kernel_size, offset, loc, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, kernel_size, set_west_or_east_u!,   u, offset,           1, bc, grid, Tuple(args); dependencies=dep, kwargs...)
@inline   fill_east_halo!(u, bc::OBC, kernel_size, offset, loc, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, kernel_size, set_west_or_east_u!,   u, offset, grid.Nx + 1, bc, grid, Tuple(args); dependencies=dep, kwargs...)
@inline  fill_south_halo!(v, bc::OBC, kernel_size, offset, loc, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, kernel_size, set_south_or_north_v!, v, offset,           1, bc, grid, Tuple(args); dependencies=dep, kwargs...)
@inline  fill_north_halo!(v, bc::OBC, kernel_size, offset, loc, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, kernel_size, set_south_or_north_v!, v, offset, grid.Ny + 1, bc, grid, Tuple(args); dependencies=dep, kwargs...)
@inline fill_bottom_halo!(w, bc::OBC, kernel_size, offset, loc, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, kernel_size, set_bottom_or_top_w!,  w, offset,           1, bc, grid, Tuple(args); dependencies=dep, kwargs...)
@inline    fill_top_halo!(w, bc::OBC, kernel_size, offset, loc, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, kernel_size, set_bottom_or_top_w!,  w, offset, grid.Nz + 1, bc, grid, Tuple(args); dependencies=dep, kwargs...)

@inline   _fill_west_halo!(j, k, grid, c, bc::OBC, loc, args...) = @inbounds c[1, j, k]           = getbc(bc, j, k, grid, args...)
@inline   _fill_east_halo!(j, k, grid, c, bc::OBC, loc, args...) = @inbounds c[grid.Nx + 1, j, k] = getbc(bc, j, k, grid, args...)
@inline  _fill_south_halo!(i, k, grid, c, bc::OBC, loc, args...) = @inbounds c[i, 1, k]           = getbc(bc, i, k, grid, args...)
@inline  _fill_north_halo!(i, k, grid, c, bc::OBC, loc, args...) = @inbounds c[i, grid.Ny + 1, k] = getbc(bc, i, k, grid, args...)
@inline _fill_bottom_halo!(i, j, grid, c, bc::OBC, loc, args...) = @inbounds c[i, j, 1]           = getbc(bc, i, j, grid, args...)
@inline    _fill_top_halo!(i, j, grid, c, bc::OBC, loc, args...) = @inbounds c[i, j, grid.Nz + 1] = getbc(bc, i, j, grid, args...)
