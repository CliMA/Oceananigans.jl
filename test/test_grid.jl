function testgridsize()
  g = RegularCartesianGrid((4, 6, 8), (2π, 4π, 9π))
  ( 4 == g.nx &&  6 == g.ny &&  8 == g.nz && 
   2π == g.Lx && 4π == g.Ly && 9π == g.Lz )
end

function testcellvolume()
  nx, ny, nz = 4, 6, 8
  dx, dy, dz = 0.1, 0.2, 0.3
  V = dx*dy*dz
  Lx, Ly, Lz = nx*dx, ny*dy, nz*dz
  g = RegularCartesianGrid((nx, ny, nz), (Lx, Ly, Lz)) 
  V == g.V
end

const d = 0.1
const n = 4

function testdx()
  g = RegularCartesianGrid((n, 2, 2), (d*n, 1, 1))
  d == g.dx
end

function testdy()
  g = RegularCartesianGrid((2, n, 2), (1, d*n, 1))
  d == g.dy
end

function testdz()
  g = RegularCartesianGrid((2, 2, n), (1, 1, d*n))
  d == g.dz
end
