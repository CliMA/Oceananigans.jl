function test_grid_size(mm::ModelMetadata)
    g = RegularCartesianGrid(mm, (4, 6, 8), (2π, 4π, 9π))

    # Checking ≈ as the grid could be storing Float32 values.
    (g.Nx ≈ 4  && g.Ny ≈ 6  && g.Nz ≈ 8 &&
     g.Lx ≈ 2π && g.Ly ≈ 4π && g.Lz ≈ 9π)
end

function test_cell_volume(mm::ModelMetadata)
    Nx, Ny, Nz = 19, 13, 7
    Δx, Δy, Δz = 0.1, 0.2, 0.3
    Lx, Ly, Lz = Nx*Δx, Ny*Δy, Nz*Δz
    g = RegularCartesianGrid(mm, (Nx, Ny, Nz), (Lx, Ly, Lz))

    # Checking ≈ as the grid could be storing Float32 values.
    g.V ≈ Δx*Δy*Δz
end

function test_faces_start_at_zero(mm::ModelMetadata)
    g = RegularCartesianGrid(mm, (10, 10, 10), (2π, 2π, 2π))
    g.xF[1] == 0 && g.yF[1] == 0 && g.zF[1] == 0
end
