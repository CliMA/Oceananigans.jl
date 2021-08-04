#####
##### Periodic boundary conditions
#####

using KernelAbstractions.Extras.LoopInfo: @unroll

@kernel function fill_periodic_west_and_east_halo!(c, H::Int, N)
    j, k = @index(Global, NTuple)
    @unroll for i = 1:H
        @inbounds begin
            c[i, j, k] = c[N+i, j, k]     # west
            c[N+H+i, j, k] = c[H+i, j, k] # east
        end
    end
end

@kernel function fill_periodic_south_and_north_halo!(c, H::Int, N)
    i, k = @index(Global, NTuple)
    @unroll for j = 1:H
        @inbounds begin
            c[i, j, k] = c[i, N+j, k]     # south
            c[i, N+H+j, k] = c[i, H+j, k] # north
        end
    end
end

@kernel function fill_periodic_bottom_and_top_halo!(c, H::Int, N)
    i, j = @index(Global, NTuple)
    @unroll for k = 1:H
        @inbounds begin
            c[i, j, k] = c[i, j, N+k]        # top
            c[i, j, k, N+H+k] = c[i, j, H+k] # bottom
        end
    end
end

fill_west_and_east_halo!(c, ::PBC, ::PBC, arch, dep, grid, args...; kw...) =
    (NoneEvent(), launch!(arch, grid, :yz, fill_periodic_west_and_east_halo!, c, grid.Hx, grid.Nx; dependencies=dep, kw...))

fill_south_and_north_halo!(c, ::PBC, ::PBC, arch, dep, grid, args...; kw...) =
    (NoneEvent(), launch!(arch, grid, :xz, fill_periodic_south_and_north_halo!, c, grid.Hy, grid.Ny; dependencies=dep, kw...))

fill_bottom_and_top_halo!(c, ::PBC, ::PBC, arch, dep, grid, args...; kw...) =
    (NoneEvent(), launch!(arch, grid, :xy, fill_periodic_bottom_and_top_halo!, c, grid.Hz, grid.Nz; dependencies=dep, kw...))
