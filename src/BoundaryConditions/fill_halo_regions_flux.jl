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

@inline function _fill_west_halo!(j, k, c, ::FBC, grid)
    @unroll for i in 1:grid.Hx
        _fill_flux_west_halo!(i, j, k, grid, c)
    end
end

@inline function _fill_east_halo!(j, k, c, ::FBC, grid)
    @unroll for i in 1:grid.Hx
        _fill_flux_east_halo!(i, j, k, grid, c)
    end
end

@inline function _fill_south_halo!(i, k, c, ::FBC, grid)
    @unroll for j in 1:grid.Hy
        _fill_flux_south_halo!(i, j, k, grid, c)
    end
end

@inline function _fill_north_halo!(i, k, c, ::FBC, grid)
    @unroll for j in 1:grid.Hy
        _fill_flux_north_halo!(i, j, k, grid, c)
    end
end

@inline function _fill_bottom_halo!(i, j, c, ::FBC, grid)
    @unroll for k in 1:grid.Hz
        _fill_flux_bottom_halo!(i, j, k, grid, c)
    end
end

@inline function _fill_top_halo!(i, j, c, ::FBC, grid)
    @unroll for k in 1:grid.Hz
        _fill_flux_top_halo!(i, j, k, grid, c)
    end
end

#####
##### Single halo filling kernels
#####

@kernel function fill_flux_west_halo!(c, grid)
    j, k = @index(Global, NTuple)

    @unroll for i in 1:grid.Hx
        _fill_flux_west_halo!(i, j, k, grid, c)
    end
end

@kernel function fill_flux_south_halo!(c, grid)
    i, k = @index(Global, NTuple)

    @unroll for j in 1:grid.Hy
        _fill_flux_south_halo!(i, j, k, grid, c)
    end
end

@kernel function fill_flux_bottom_halo!(c, grid)
    i, j = @index(Global, NTuple)

    @unroll for k in 1:grid.Hz
        _fill_flux_bottom_halo!(i, j, k, grid, c)
    end
end

@kernel function fill_flux_east_halo!(c, grid)
    j, k = @index(Global, NTuple)

    @unroll for i in 1:grid.Hx
        _fill_flux_east_halo!(i, j, k, grid, c)
    end
end

@kernel function fill_flux_north_halo!(c, grid)
    i, k = @index(Global, NTuple)

    @unroll for j in 1:grid.Hy
        _fill_flux_north_halo!(i, j, k, grid, c)
    end
end

@kernel function fill_flux_top_halo!(c, grid)
    i, j = @index(Global, NTuple)

    @unroll for k in 1:grid.Hz
        _fill_flux_top_halo!(i, j, k, grid, c)
    end
end

#####
##### Kernel launchers for flux boundary conditions
#####

  fill_west_halo!(c, bc::FBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :yz, fill_flux_west_halo!,   c, grid; dependencies=dep, kwargs...)
  fill_east_halo!(c, bc::FBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :yz, fill_flux_east_halo!,   c, grid; dependencies=dep, kwargs...)
 fill_south_halo!(c, bc::FBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xz, fill_flux_south_halo!,  c, grid; dependencies=dep, kwargs...)
 fill_north_halo!(c, bc::FBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xz, fill_flux_north_halo!,  c, grid; dependencies=dep, kwargs...)
fill_bottom_halo!(c, bc::FBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xy, fill_flux_bottom_halo!, c, grid; dependencies=dep, kwargs...)
   fill_top_halo!(c, bc::FBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xy, fill_flux_top_halo!,    c, grid; dependencies=dep, kwargs...)
