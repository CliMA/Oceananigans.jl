#####
##### Outer functions for filling halo regions for open boundary conditions.
#####

# TODO: support true open boundary conditions.
# For this we need to have separate functions for each of the six boundaries,
# and need to unroll a loop over the boundary normal direction.
# The syntax for `getbc` is also different for OpenBoundaryCondition than for others,
# because the boundary-normal index can vary (and array boundary conditions need to be
# 3D in general).

#=
@kernel function set_west_or_east_u!(u, i_boundary, bc, loc, grid, args...)
    j, k = @index(Global, NTuple)
    @inbounds u[i_boundary, j, k] = getbc(bc, j, k, grid, args...)
end

@kernel function set_south_or_north_v!(v, j_boundary, bc, loc, grid, args...)
    i, k = @index(Global, NTuple)
    @inbounds v[i, j_boundary, k] = getbc(bc, i, k, grid, args...)
end

@kernel function set_bottom_or_top_w!(w, k_boundary, bc, loc, grid, args...)
    i, j = @index(Global, NTuple)
    @inbounds w[i, j, k_boundary] = getbc(bc, i, j, grid, args...)
end

  @inline fill_west_halo!(u, bc::OBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :yz, set_west_or_east_u!,   u,           1, bc, grid, args...; dependencies=dep, kwargs...)
  @inline fill_east_halo!(u, bc::OBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :yz, set_west_or_east_u!,   u, grid.Nx + 1, bc, grid, args...; dependencies=dep, kwargs...)
 @inline fill_south_halo!(v, bc::OBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xz, set_south_or_north_v!, v,           1, bc, grid, args...; dependencies=dep, kwargs...)
 @inline fill_north_halo!(v, bc::OBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xz, set_south_or_north_v!, v, grid.Ny + 1, bc, grid, args...; dependencies=dep, kwargs...)
@inline fill_bottom_halo!(w, bc::OBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xy, set_bottom_or_top_w!,  w,           1, bc, grid, args...; dependencies=dep, kwargs...)
   @inline fill_top_halo!(w, bc::OBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xy, set_bottom_or_top_w!,  w, grid.Nz + 1, bc, grid, args...; dependencies=dep, kwargs...)
=#

@inline   _fill_west_halo!(j, k, grid, c, bc::OBC, args...) = @inbounds c[1, j, k]           = getbc(bc, j, k, grid, args...)
@inline   _fill_east_halo!(j, k, grid, c, bc::OBC, args...) = @inbounds c[grid.Nx + 1, j, k] = getbc(bc, j, k, grid, args...)
@inline  _fill_south_halo!(i, k, grid, c, bc::OBC, args...) = @inbounds c[i, 1, k]           = getbc(bc, i, k, grid, args...)
@inline  _fill_north_halo!(i, k, grid, c, bc::OBC, args...) = @inbounds c[i, grid.Ny + 1, k] = getbc(bc, i, k, grid, args...)
@inline _fill_bottom_halo!(i, j, grid, c, bc::OBC, args...) = @inbounds c[i, j, 1]           = getbc(bc, i, j, grid, args...)
@inline    _fill_top_halo!(i, j, grid, c, bc::OBC, args...) = @inbounds c[i, j, grid.Nz + 1] = getbc(bc, i, j, grid, args...)
