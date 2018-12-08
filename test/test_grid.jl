function test_grid_size()
    g = RegularCartesianGrid((4, 6, 8), (2π, 4π, 9π))
    (g.Nx == 4  && g.Ny == 6  && g.Nz == 8 &&
    g.Lx == 2π && g.Ly == 4π && g.Lz == 9π)
end

function test_cell_volume()
    Nx, Ny, Nz = 4, 6, 8
    Δx, Δy, Δz = 0.1, 0.2, 0.3
    V = Δx*Δy*Δz
    Lx, Ly, Lz = Nx*Δx, Ny*Δy, Nz*Δz
    g = RegularCartesianGrid((Nx, Ny, Nz), (Lx, Ly, Lz))
    V == g.V
end

function test_Δx(d, n)
    g = RegularCartesianGrid((n, 2, 2), (d*n, 1, 1))
    d == g.Δx
end

function test_Δy(d, n)
    g = RegularCartesianGrid((2, n, 2), (1, d*n, 1))
    d == g.Δy
end

function test_Δz(d, n)
    g = RegularCartesianGrid((2, 2, n), (1, 1, d*n))
    d == g.Δz
end

function test_faces_start_at_zero()
    g = RegularCartesianGrid((10, 10, 10), (2π, 2π, 2π))
    g.xF[1] == 0 && g.yF[1] == 0 && g.zF[1] == 0
end
