function testgridsize()
  g = Grid((4, 6, 8), (2π, 4π, 9π))
  ( 4 == g.nx &&  6 == g.ny &&  8 == g.nz && 
   2π == g.Lx && 4π == g.Ly && 9π == g.Lz )
end

function testcellvolume()
  dx = 0.1
  dy = 0.2
  dz = 0.3
  V = dx*dy*dz
  nx = 4
  ny = 6
  nz = 8
  Lx = nx*dx
  Ly = ny*dy
  Lz = nz*dz
  g = Grid((nx, ny, nz), (Lx, Ly, Lz)) 
  V == g.V
end

function testdx()
  dx, nx = 0.1, 4
  Lx = nx*dx
  g = Grid((nx, 2, 2), (Lx, 1, 1))
  dx == g.dx
end

function testdy()
  dy, ny = 0.6, 6
  Ly = ny*dy
  g = Grid((2, ny, 2), (1, Ly, 1))
  dy == g.dy
end

function testdz()
  dz, nz = 0.3, 10
  Lz = nz*dz
  g = Grid((2, 2, nz), (1, 1, Lz))
  dz == g.dz
end
