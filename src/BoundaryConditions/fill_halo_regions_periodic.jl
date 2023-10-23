using KernelAbstractions.Extras.LoopInfo: @unroll

#####
##### Periodic boundary conditions
#####

@inline parent_size_and_offset(c, dim1, dim2, size, offset)     = (parent(c), size, fix_halo_offsets.(offset, c.offsets[[dim1, dim2]]))
@inline parent_size_and_offset(c, dim1, dim2, ::Symbol, offset) = (parent(c), size(parent(c))[[dim1, dim2]], (0, 0))

@inline function parent_size_and_offset(c::NTuple, dim1, dim2, ::Symbol, offset)
    p = parent.(c)
    p_size = (minimum([size(t, dim1) for t in p]), minimum([size(t, dim2) for t in p]))
    return p, p_size, (0, 0)
end

@inline fix_halo_offsets(o, co) = co > 0 ? o - co : o # Windowed fields have only positive offsets to correct

function fill_west_and_east_halo!(c, ::PBCT, ::PBCT, size, offset, loc, arch, grid, args...; kw...)
    c_parent, yz_size, offset = parent_size_and_offset(c, 2, 3, size, offset)
    launch!(arch, grid, KernelParameters(yz_size, offset), fill_periodic_west_and_east_halo!, c_parent, grid.Hx, grid.Nx; kw...)
    return nothing
end

function fill_south_and_north_halo!(c, ::PBCT, ::PBCT, size, offset, loc, arch, grid, args...; kw...)
    c_parent, xz_size, offset = parent_size_and_offset(c, 1, 3, size, offset)
    launch!(arch, grid, KernelParameters(xz_size, offset), fill_periodic_south_and_north_halo!, c_parent, grid.Hy, grid.Ny;  kw...)
    return nothing
end

function fill_bottom_and_top_halo!(c, ::PBCT, ::PBCT, size, offset, loc, arch, grid, args...; kw...)
    c_parent, xy_size, offset = parent_size_and_offset(c, 1, 2, size, offset)
    launch!(arch, grid, KernelParameters(xy_size, offset), fill_periodic_bottom_and_top_halo!, c_parent, grid.Hz, grid.Nz; kw...)
    return nothing
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
            c[i, j, k]     = c[i, N+j, k] # south
            c[i, N+H+j, k] = c[i, H+j, k] # north
        end
    end
end

@kernel function fill_periodic_bottom_and_top_halo!(c, H::Int, N)
    i, j = @index(Global, NTuple)
    @unroll for k = 1:H
        @inbounds begin
            c[i, j, k]     = c[i, j, N+k] # top
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

#####
##### Throw error if single-sided periodic boundary conditions are used
#####

fill_west_halo!(c, ::PBCT, args...; kwargs...)   = throw(ArgumentError("Periodic boundary conditions must be applied to both sides"))
fill_east_halo!(c, ::PBCT, args...; kwargs...)   = throw(ArgumentError("Periodic boundary conditions must be applied to both sides"))
fill_south_halo!(c, ::PBCT, args...; kwargs...)  = throw(ArgumentError("Periodic boundary conditions must be applied to both sides"))
fill_north_halo!(c, ::PBCT, args...; kwargs...)  = throw(ArgumentError("Periodic boundary conditions must be applied to both sides"))
fill_bottom_halo!(c, ::PBCT, args...; kwargs...) = throw(ArgumentError("Periodic boundary conditions must be applied to both sides"))
fill_top_halo!(c, ::PBCT, args...; kwargs...)    = throw(ArgumentError("Periodic boundary conditions must be applied to both sides"))
