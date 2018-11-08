struct RegularCartesianGrid{T<:AbstractFloat}
  # Number of grid points in (x,y,z).
  nx::Int
  ny::Int
  nz::Int
  # Domain size [m].
  Lx::T
  Ly::T
  Lz::T
  # Grid spacing [m].
  dx::T
  dy::T
  dz::T
  # Cell face areas [m²].
  Ax::T
  Ay::T
  Az::T
  # Volume of a cell [m³].
  V::T
end

function RegularCartesianGrid(n, L; T=Float64)
  nx, ny, nz = n
  Lx, Ly, Lz = L

  dx = Lx / nx
  dy = Ly / ny
  dz = Lz / nz

  Ax = dx*dz
  Ay = dx*dz
  Az = dx*dy

  V = dx*dy*dz

  RegularCartesianGrid{T}(nx, ny, nz, Lx, Ly, Lz, dx, dy, dz, Ax, Ay, Az)
end

# example: g = RegularCartesianGrid((16, 16, 8), (2π, 2π, 2π))
