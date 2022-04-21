using KernelAbstractions.Extras.LoopInfo: @unroll

#####
##### Periodic boundary conditions
#####

@inline parent_and_size(c, dim1, dim2) = (parent(c),  size(parent(c))[[dim1, dim2]])

@inline function parent_and_size(c::NTuple, dim1, dim2)
    p = parent.(c)
    p_size = (minimum([size(t, dim1) for t in p]), minimum([size(t, dim2) for t in p]))
    return p, p_size
end

function fill_west_and_east_halo!(c, ::PBCT, ::PBCT, loc, arch, dep, grid, args...; kw...)
    c_parent, yz_size = parent_and_size(c, 2, 3)
    event = launch!(arch, grid, yz_size, fill_periodic_west_and_east_halo!, c_parent, grid.Hx, grid.Nx; dependencies=dep, kw...)
    return event
end

function fill_south_and_north_halo!(c, ::PBCT, ::PBCT, loc, arch, dep, grid, args...; kw...)
    c_parent, xz_size = parent_and_size(c, 1, 3)
    event = launch!(arch, grid, xz_size, fill_periodic_south_and_north_halo!, c_parent, grid.Hy, grid.Ny; dependencies=dep, kw...)
    return event
end

function fill_bottom_and_top_halo!(c, ::PBCT, ::PBCT, loc, arch, dep, grid, args...; kw...)
    c_parent, xy_size = parent_and_size(c, 1, 2)
    event = launch!(arch, grid, xy_size, fill_periodic_bottom_and_top_halo!, c_parent, grid.Hz, grid.Nz; dependencies=dep, kw...)
    return event
end

#####
##### Periodic boundary condition kernels
#####

@kernel function fill_periodic_west_and_east_halo!(c, H::Int, N)
    j, k = @index(Global, NTuple)
    @unroll for i = 1:H
        @inbounds begin
            c[i, j, k]     = c[N+i, j, k] # west
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
            c[i, j, N+H+k] = c[i, j, H+k] # bottom
        end
    end
end

####
#### Tupled periodic boundary condition 
####

@kernel function fill_periodic_west_and_east_halo!(c::NTuple{M}, H::Int, N) where M
    j, k = @index(Global, NTuple)
    @unroll for n = 1:M
        @unroll for i = 1:H
            @inbounds begin
                  c[n][i, j, k]     = c[n][N+i, j, k] # west
                  c[n][N+H+i, j, k] = c[n][H+i, j, k] # east
            end
        end
    end
end

@kernel function fill_periodic_south_and_north_halo!(c::NTuple{M}, H::Int, N) where M
    i, k = @index(Global, NTuple)
    @unroll for n = 1:M
        @unroll for j = 1:H
            @inbounds begin
                c[n][i, j, k]     = c[n][i, N+j, k] # south
                c[n][i, N+H+j, k] = c[n][i, H+j, k] # north
            end
        end
    end
end

@kernel function fill_periodic_bottom_and_top_halo!(c::NTuple{M}, H::Int, N) where M
    i, j = @index(Global, NTuple)
    @unroll for n = 1:M
        @unroll for k = 1:H
            @inbounds begin
                c[n][i, j, k]     = c[n][i, j, N+k] # top
                c[n][i, j, N+H+k] = c[n][i, j, H+k] # bottom
            end  
        end
    end
end
