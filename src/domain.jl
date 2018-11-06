struct Grid{T<:AbstractFloat}
  nx::Int
  ny::Int
  nz::Int
  Lx::T
  Ly::T
  Lz::T
  dx::T
  dy::T
  dz::T
  Ax::T
  Ay::T
  Az::T
  V::T
end

function Grid(n, L; T=Float64)
  nx, ny, nz = n              
  Lx, Ly, Lz = L              

  dx = Lx/nx
  dy = Ly/ny
  dz = Lz/nz

  Ax = dx*dz
  Ay = dx*dz
  Az = dx*dy

  V = dx*dy*dz

  Grid{T}(nx, ny, nz, Lx, Ly, Lz, dx, dy, dz, Ax, Ay, Az)
end

# example: g = Grid((16, 16, 8), (2π, 2π, 2π))
