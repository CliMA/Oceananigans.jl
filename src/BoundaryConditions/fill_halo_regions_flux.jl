using KernelAbstractions.Extras.LoopInfo: @unroll

#####
##### Kernels that ensure 'no-flux' from second- and fourth-order diffusion operators.
##### Note that flux divergence associated with a flux boundary condition is added
##### in a separate step.
#####

@inline fill_flux_west_halo!(c, ::FBC, N, i, j, k) = @inbounds c[1-i, j, k] = c[i, j, k]
@inline fill_flux_east_halo!(c, ::FBC, N, i, j, k) = @inbounds c[N+i, j, k] = c[N+1-i, j, k]

@inline fill_flux_south_halo!(c, ::FBC, N, i, j, k) = @inbounds c[i, 1-j, k] = c[i, j, k]
@inline fill_flux_north_halo!(c, ::FBC, N, i, j, k) = @inbounds c[i, N+j, k] = c[i, N+1-j, k]

@inline fill_flux_bottom_halo!(c, ::FBC, N, i, j, k) = @inbounds c[i, j, 1-k] = c[i, j, k]
@inline fill_flux_top_halo!(c,    ::FBC, N, i, j, k) = @inbounds c[i, j, N+k] = c[i, j, N+1-k]

@kernel function fill_flux_west_and_east_halo!(c, west_bc, east_bc, Hx, Nx)
    j, k = @index(Global, NTuple)

    @unroll for i in 1:Hx
        fill_flux_west_halo!(c, west_bc, Nx, i, j, k)
        fill_flux_east_halo!(c, east_bc, Nx, i, j, k)
    end
end

@kernel function fill_flux_south_and_north_halo!(c, south_bc, north_bc, Hy, Ny)
    i, k = @index(Global, NTuple)

    @unroll for j in 1:Hy
        fill_flux_south_halo!(c, south_bc, Ny, i, j, k)
        fill_flux_north_halo!(c, north_bc, Ny, i, j, k)
    end
end

@kernel function fill_flux_bottom_and_top_halo!(c, bottom_bc, top_bc, Hz, Nz)
    i, j = @index(Global, NTuple)

    @unroll for k in 1:Hz
        fill_flux_bottom_halo!(c, bottom_bc, Nz, i, j, k)
        fill_flux_top_halo!(c, top_bc, Nz, i, j, k)
    end
end

fill_west_and_east_halo!(c, west_bc::FBC, east_bc::FBC, arch, dep, grid, args...; kwargs...) =
    (NoneEvent(), launch!(arch, grid, :yz, fill_flux_west_and_east_halo!, c, west_bc, east_bc, grid.Hx, grid.Nx; dependencies=dep, kwargs...))

fill_south_and_north_halo!(c, south_bc::FBC, north_bc::FBC, arch, dep, grid, args...; kwargs...) =
    (NoneEvent(), launch!(arch, grid, :xz, fill_flux_south_and_north_halo!, c, south_bc, north_bc, grid.Hy, grid.Ny; dependencies=dep, kwargs...))

fill_bottom_and_top_halo!(c, bottom_bc::FBC, top_bc::FBC, arch, dep, grid, args...; kwargs...) =
    (NoneEvent(), launch!(arch, grid, :xy, fill_flux_bottom_and_top_halo!, c, bottom_bc, top_bc, grid.Hz, grid.Nz; dependencies=dep, kwargs...))

