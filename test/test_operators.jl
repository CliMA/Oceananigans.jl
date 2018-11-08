function testxderiv()
  nx, Lx = 1024, 2π 
  ny, nz = 2, 2
  Ly, Lz = 1, 1
  k = 2
  g = Grid((nx, ny, nz), (Lx, Ly, Lz))

    u = zeros(nx, ny, nz)
   ux = zeros(nx, ny, nz)
  uxx = zeros(nx, ny, nz)
  uxx_analytic = zeros(nx, ny, nz)
  
  @. u = sin(k*g.x)
  xderiv!(ux, u, g)
  xderivplus!(uxx, ux, g)

  @. uxx_analytic = -k^2*sin(k*g.x)

  isapprox(uxx, uxx_analytic, rtol=1e-4)
end
