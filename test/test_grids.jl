function test_grid_size(ft::DataType)
    g = RegularCartesianGrid(ft, (4, 6, 8), (2π, 4π, 9π))

    # Checking ≈ as the grid could be storing Float32 values.
    (g.Nx ≈ 4  && g.Ny ≈ 6  && g.Nz ≈ 8 &&
     g.Lx ≈ 2π && g.Ly ≈ 4π && g.Lz ≈ 9π)
end

function test_cell_volume(ft::DataType)
    Nx, Ny, Nz = 19, 13, 7
    Δx, Δy, Δz = 0.1, 0.2, 0.3
    Lx, Ly, Lz = Nx*Δx, Ny*Δy, Nz*Δz
    g = RegularCartesianGrid(ft, (Nx, Ny, Nz), (Lx, Ly, Lz))

    # Checking ≈ as the grid could be storing Float32 values.
    g.V ≈ Δx*Δy*Δz
end

function test_faces_start_at_zero(ft::DataType)
    g = RegularCartesianGrid(ft, (10, 10, 10), (2π, 2π, 2π))
    g.xF[1] == 0 && g.yF[1] == 0 && g.zF[1] == 0
end

function test_end_faces_match_grid_length(ft::DataType)
    g = RegularCartesianGrid(ft, (12, 13, 14), (π, π^2, π^3))
    (g.xF[end] - g.xF[1] ≈ π   &&
     g.yF[end] - g.yF[1] ≈ π^2 &&
     g.zF[1] - g.zF[end] ≈ π^3)
end
