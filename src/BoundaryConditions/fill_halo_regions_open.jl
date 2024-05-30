#####
##### Outer functions for filling halo regions for open boundary conditions.
#####

# TODO: support true open boundary conditions.
# For this we need to have separate functions for each of the six boundaries,
# and need to unroll a loop over the boundary normal direction.
# The syntax for `getbc` is also different for OpenBoundaryCondition than for others,
# because the boundary-normal index can vary (and array boundary conditions need to be
# 3D in general).

@kernel function set_west_or_east_u!(u, i_boundary, bc, grid, args) 
    j, k = @index(Global, NTuple)
    @inbounds u[i_boundary, j, k] = getbc(bc, j, k, grid, args...)
end

@kernel function set_south_or_north_v!(v, j_boundary, bc, grid, args)
    i, k = @index(Global, NTuple)
    @inbounds v[i, j_boundary, k] = getbc(bc, i, k, grid, args...)
end

@kernel function set_bottom_or_top_w!(w, k_boundary, bc, grid, args) 
    i, j = @index(Global, NTuple)
    @inbounds w[i, j, k_boundary] = getbc(bc, i, j, grid, args...)
end

@inline   _fill_west_halo!(j, k, grid, c, bc::OBC, loc, pressure_corrected, args...) = @inbounds c[1, j, k]           = getbc(bc, j, k, grid, args...)
@inline   _fill_east_halo!(j, k, grid, c, bc::OBC, loc, pressure_corrected, args...) = @inbounds c[grid.Nx + 1, j, k] = getbc(bc, j, k, grid, args...)
@inline  _fill_south_halo!(i, k, grid, c, bc::OBC, loc, pressure_corrected, args...) = @inbounds c[i, 1, k]           = getbc(bc, i, k, grid, args...)
@inline  _fill_north_halo!(i, k, grid, c, bc::OBC, loc, pressure_corrected, args...) = @inbounds c[i, grid.Ny + 1, k] = getbc(bc, i, k, grid, args...)
@inline _fill_bottom_halo!(i, j, grid, c, bc::OBC, loc, pressure_corrected, args...) = @inbounds c[i, j, 1]           = getbc(bc, i, j, grid, args...)
@inline    _fill_top_halo!(i, j, grid, c, bc::OBC, loc, pressure_corrected, args...) = @inbounds c[i, j, grid.Nz + 1] = getbc(bc, i, j, grid, args...)

# refuse if wall normal

@inline   _fill_west_halo!(j, k, grid::AbstractGrid{<:Any, Bounded}, c, bc::OBC, loc::Tuple{Face, Center, Center}, pressure_corrected::Val{true}, args...) = nothing
@inline   _fill_east_halo!(j, k, grid::AbstractGrid{<:Any, Bounded}, c, bc::OBC, loc::Tuple{Face, Center, Center}, pressure_corrected::Val{true}, args...) = nothing
@inline   _fill_south_halo!(j, k, grid::AbstractGrid{<:Any, <:Any, Bounded}, c, bc::OBC, loc::Tuple{Center, Face, Center}, pressure_corrected::Val{true}, args...) = nothing
@inline   _fill_north_halo!(j, k, grid::AbstractGrid{<:Any, <:Any, Bounded}, c, bc::OBC, loc::Tuple{Center, Face, Center}, pressure_corrected::Val{true}, args...) = nothing
@inline   _fill_bottom_halo!(j, k, grid::AbstractGrid{<:Any, <:Any, <:Any, Bounded}, c, bc::OBC, loc::Tuple{Center, Center, Face}, pressure_corrected::Val{true}, args...) = nothing
@inline   _fill_top_halo!(j, k, grid::AbstractGrid{<:Any, <:Any, <:Any, Bounded}, c, bc::OBC, loc::Tuple{Center, Center, Face}, pressure_corrected::Val{true}, args...) = nothing