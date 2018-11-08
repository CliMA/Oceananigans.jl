"""
    Grid([T=Float64], n, L)

Construct a `Grid` with resolution `n = (nx, ny, nz)` and size `L = (Lx, Ly, Lz)`.
Optionally specify the type of the grid with `T`.
"""
struct Grid{T,Tx}
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
  x::Tx
  y::Tx
  z::Tx
end

function Grid(n, L, T=Float64)
  nx, ny, nz = n              
  Lx, Ly, Lz = L              

  dx = Lx/nx
  dy = Ly/ny
  dz = Lz/nz

  Ax = dx*dz
  Ay = dx*dz
  Az = dx*dy

  V = dx*dy*dz

  x = Array{T,3}(reshape(range(0, length=nx, step=dx), (nx, 1, 1)))
  y = Array{T,3}(reshape(range(0, length=ny, step=dy), (1, ny, 1)))
  z = Array{T,3}(reshape(range(0, length=nz, step=dz), (1, 1, nz)))

  Grid{T,Array{T,3}}(nx, ny, nz, Lx, Ly, Lz, dx, dy, dz, Ax, Ay, Az, V, x, y, z)
end

Grid(T::DataType, n, L) = Grid(n, L, T)
