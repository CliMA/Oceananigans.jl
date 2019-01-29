function test_grid_size()
    g = RegularCartesianGrid((4, 6, 8), (2π, 4π, 9π); FloatType=Float64)

    (g.Nx == 4  && g.Ny == 6  && g.Nz == 8 &&
     g.Lx == 2π && g.Ly == 4π && g.Lz == 9π)
end

function test_cell_volume()
    Nx, Ny, Nz = 19, 13, 7
    Δx, Δy, Δz = 0.1, 0.2, 0.3
    V = Δx*Δy*Δz
    Lx, Ly, Lz = Nx*Δx, Ny*Δy, Nz*Δz
    g = RegularCartesianGrid((Nx, Ny, Nz), (Lx, Ly, Lz))

    V == g.V
end

function test_faces_start_at_zero()
    g = RegularCartesianGrid((10, 10, 10), (2π, 2π, 2π); FloatType=Float64)
    g.xF[1] == 0 && g.yF[1] == 0 && g.zF[1] == 0
end
