using KernelAbstractions.Extras.LoopInfo: @unroll

#####
##### Periodic boundary conditions
#####

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
          c[i, j, N+H+k] = c[i, j, H+k] # bottom
      end
  end
end

function fill_west_and_east_halo!(c, ::PBC, ::PBC, arch, dep, grid, args...; kw...)
  c_parent = parent(c)
  yz_size = size(c_parent)[[2, 3]]
  event = launch!(arch, grid, yz_size, fill_periodic_west_and_east_halo!, c_parent, grid.Hx, grid.Nx; dependencies=dep, kw...)
  return event
end

function fill_south_and_north_halo!(c, ::PBC, ::PBC, arch, dep, grid, args...; kw...)
  c_parent = parent(c)
  xz_size = size(c_parent)[[1, 3]]
  event = launch!(arch, grid, xz_size, fill_periodic_south_and_north_halo!, c_parent, grid.Hy, grid.Ny; dependencies=dep, kw...)
  return event
end

function fill_bottom_and_top_halo!(c, ::PBC, ::PBC, arch, dep, grid, args...; kw...)
  c_parent = parent(c)
  xy_size = size(c_parent)[[1, 2]]
  event = launch!(arch, grid, xy_size, fill_periodic_bottom_and_top_halo!, c_parent, grid.Hz, grid.Nz; dependencies=dep, kw...)
  return event
end

####
#### Implement single periodic directions (for Distributed memory architectures)
####


@kernel function fill_periodic_west_halo!(c, H::Int, N)
  j, k = @index(Global, NTuple)
  @unroll for i = 1:H
      @inbounds begin
          c[i, j, k] = c[N+i, j, k]     # west
      end
  end
end

@kernel function fill_periodic_east_halo!(c, H::Int, N)
  j, k = @index(Global, NTuple)
  @unroll for i = 1:H
      @inbounds begin
          c[N+H+i, j, k] = c[H+i, j, k] # east
      end
  end
end

@kernel function fill_periodic_south_halo!(c, H::Int, N)
  i, k = @index(Global, NTuple)
  @unroll for j = 1:H
      @inbounds begin
          c[i, j, k] = c[i, N+j, k]     # south
      end
  end
end

@kernel function fill_periodic_north_halo!(c, H::Int, N)
  i, k = @index(Global, NTuple)
  @unroll for j = 1:H
      @inbounds begin
          c[i, N+H+j, k] = c[i, H+j, k] # north
      end
  end
end

@kernel function fill_periodic_bottom_halo!(c, H::Int, N)
  i, j = @index(Global, NTuple)
  @unroll for k = 1:H
      @inbounds begin
          c[i, j, k] = c[i, j, N+k]        # bottom
      end
  end
end

@kernel function fill_periodic_top_halo!(c, H::Int, N)
  i, j = @index(Global, NTuple)
  @unroll for k = 1:H
      @inbounds begin
          c[i, j, N+H+k] = c[i, j, H+k] # top
      end
  end
end

function fill_west_halo!(c, ::PBC, arch, dep, grid, args...; kw...)
  c_parent = parent(c)
  yz_size = size(c_parent)[[2, 3]]
  event = launch!(arch, grid, yz_size, fill_periodic_west_halo!, c_parent, grid.Hx, grid.Nx; dependencies=dep, kw...)
  return event
end

function fill_east_halo!(c, ::PBC, arch, dep, grid, args...; kw...)
  c_parent = parent(c)
  yz_size = size(c_parent)[[2, 3]]
  event = launch!(arch, grid, yz_size, fill_periodic_east_halo!, c_parent, grid.Hx, grid.Nx; dependencies=dep, kw...)
  return event
end

function fill_south_halo!(c, ::PBC, arch, dep, grid, args...; kw...)
  c_parent = parent(c)
  xz_size = size(c_parent)[[1, 3]]
  event = launch!(arch, grid, xz_size, fill_periodic_south_halo!, c_parent, grid.Hy, grid.Ny; dependencies=dep, kw...)
  return event
end

function fill_north_halo!(c, ::PBC, arch, dep, grid, args...; kw...)
  c_parent = parent(c)
  xz_size = size(c_parent)[[1, 3]]
  event = launch!(arch, grid, xz_size, fill_periodic_north_halo!, c_parent, grid.Hy, grid.Ny; dependencies=dep, kw...)
  return event
end

function fill_bottom_halo!(c, ::PBC, arch, dep, grid, args...; kw...)
  c_parent = parent(c)
  xy_size = size(c_parent)[[1, 2]]
  event = launch!(arch, grid, xy_size, fill_periodic_bottom_halo!, c_parent, grid.Hz, grid.Nz; dependencies=dep, kw...)
  return event
end

function fill_top_halo!(c, ::PBC, arch, dep, grid, args...; kw...)
  c_parent = parent(c)
  xy_size = size(c_parent)[[1, 2]]
  event = launch!(arch, grid, xy_size, fill_periodic_top_halo!, c_parent, grid.Hz, grid.Nz; dependencies=dep, kw...)
  return event
end
