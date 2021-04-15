using KernelAbstractions.Extras.LoopInfo: @unroll

#####
##### Kernels that ensure 'no-flux' from second- and fourth-order diffusion operators.
##### Note that flux divergence associated with a flux boundary condition is added
##### in a separate step.
#####

@kernel function _fill_west_halo!(c, ::FBC, H, N)
    j, k = @index(Global, NTuple)

    @unroll for i in 1:H
        @inbounds c[1-i, j, k] = c[i, j, k]
    end
end

@kernel function _fill_south_halo!(c, ::FBC, H, N)
    i, k = @index(Global, NTuple)

    @unroll for j in 1:H
        @inbounds c[i, 1-j, k] = c[i, j, k]
    end
end

@kernel function _fill_bottom_halo!(c, ::FBC, H, N)
    i, j = @index(Global, NTuple)

    @unroll for k in 1:H
        @inbounds c[i, j, 1-k] = c[i, j, k]
    end
end

@kernel function _fill_east_halo!(c, ::FBC, H, N)
    j, k = @index(Global, NTuple)

    @unroll for i in 1:H
        @inbounds c[N+i, j, k] = c[N+1-i, j, k]
    end
end

@kernel function _fill_north_halo!(c, ::FBC, H, N)
    i, k = @index(Global, NTuple)

    @unroll for j in 1:H
        @inbounds c[i, N+j, k] = c[i, N+1-j, k]
    end
end

@kernel function _fill_top_halo!(c, ::FBC, H, N)
    i, j = @index(Global, NTuple)

    @unroll for k in 1:H
        @inbounds c[i, j, N+k] = c[i, j, N+1-k]
    end
end

#####
##### Kernel launchers for flux boundary conditions
#####

  fill_west_halo!(c, bc::FBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :yz, _fill_west_halo!,   c, bc, grid.Hx, grid.Nx; dependencies=dep, kwargs...)
  fill_east_halo!(c, bc::FBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :yz, _fill_east_halo!,   c, bc, grid.Hx, grid.Nx; dependencies=dep, kwargs...)
 fill_south_halo!(c, bc::FBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xz, _fill_south_halo!,  c, bc, grid.Hy, grid.Ny; dependencies=dep, kwargs...)
 fill_north_halo!(c, bc::FBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xz, _fill_north_halo!,  c, bc, grid.Hy, grid.Ny; dependencies=dep, kwargs...)
fill_bottom_halo!(c, bc::FBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xy, _fill_bottom_halo!, c, bc, grid.Hz, grid.Nz; dependencies=dep, kwargs...)
   fill_top_halo!(c, bc::FBC, arch, dep, grid, args...; kwargs...) = launch!(arch, grid, :xy, _fill_top_halo!,    c, bc, grid.Hz, grid.Nz; dependencies=dep, kwargs...)
