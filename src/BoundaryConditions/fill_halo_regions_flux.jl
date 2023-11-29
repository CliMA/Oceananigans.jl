using KernelAbstractions.Extras.LoopInfo: @unroll

##### Kernels that ensure 'no-flux' from second- and fourth-order diffusion operators.
##### Kernel functions that ensure 'no-flux' from second- and fourth-order diffusion operators.
##### Note that flux divergence associated with a flux boundary condition is added
##### in a separate step.
#####
##### We implement two sets of kernel functions: one for filling one boundary at a time, and
##### a second that fills both boundaries at the same as a performance optimization.
#####

#####
##### Low-level functions that set data
#####

@inline _fill_flux_west_halo!(i, j, k, grid, c) = @inbounds c[1-i, j, k] = c[i, j, k]
@inline _fill_flux_east_halo!(i, j, k, grid, c) = @inbounds c[grid.Nx+i, j, k] = c[grid.Nx+1-i, j, k]

@inline _fill_flux_south_halo!(i, j, k, grid, c) = @inbounds c[i, 1-j, k] = c[i, j, k]
@inline _fill_flux_north_halo!(i, j, k, grid, c) = @inbounds c[i, grid.Ny+j, k] = c[i, grid.Ny+1-j, k]

@inline _fill_flux_bottom_halo!(i, j, k, grid, c) = @inbounds c[i, j, 1-k] = c[i, j, k]
@inline _fill_flux_top_halo!(i, j, k, grid, c)    = @inbounds c[i, j, grid.Nz+k] = c[i, j, grid.Nz+1-k]

#####
#####
##### Combined halo filling functions
#####

@inline _fill_west_halo!(j, k, grid, c, ::FBC, args...)   =   _fill_flux_west_halo!(1, j, k, grid, c)
@inline _fill_east_halo!(j, k, grid, c, ::FBC, args...)   =   _fill_flux_east_halo!(1, j, k, grid, c)
@inline _fill_south_halo!(i, k, grid, c, ::FBC, args...)  =  _fill_flux_south_halo!(i, 1, k, grid, c)
@inline _fill_north_halo!(i, k, grid, c, ::FBC, args...)  =  _fill_flux_north_halo!(i, 1, k, grid, c)
@inline _fill_bottom_halo!(i, j, grid, c, ::FBC, args...) = _fill_flux_bottom_halo!(i, j, 1, grid, c)
@inline _fill_top_halo!(i, j, grid, c, ::FBC, args...)    =    _fill_flux_top_halo!(i, j, 1, grid, c)
