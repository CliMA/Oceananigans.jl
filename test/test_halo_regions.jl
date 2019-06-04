function halo_regions_initalized_correctly(arch, FT, Nx, Ny, Nz)
    # Just choose something anisotropic to catch Δx/Δy type errors.
    Lx, Ly, Lz = 100, 200, 300

    grid = RegularCartesianGrid(FT, (Nx, Ny, Nz), (Lx, Ly, Lz))
    field = CellField(FT, arch, grid)

    # Fill the interior with random numbers.
    data(field) .= rand(FT, Nx, Ny, Nz)

    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    # The halo regions should still just contain zeros.
    (all(field.data[1-Hx:0,          :,          :] .== 0) &&
     all(field.data[Nx+1:Nx+Hx,      :,          :] .== 0) &&
     all(field.data[:,          1-Hy:0,          :] .== 0) &&
     all(field.data[:,      Ny+1:Ny+Hy,          :] .== 0) &&
     all(field.data[:,               :,     1-Hz:0] .== 0) &&
     all(field.data[:,               :, Nz+1:Nz+Hz] .== 0))
end

function halo_regions_correctly_filled(arch, FT, Nx, Ny, Nz)
    # Just choose something anisotropic to catch Δx/Δy type errors.
    Lx, Ly, Lz = 100, 200, 300

    grid = RegularCartesianGrid(FT, (Nx, Ny, Nz), (Lx, Ly, Lz))
    field = CellField(FT, arch, grid)

    data(field) .= rand(FT, Nx, Ny, Nz)
    fill_halo_regions!(arch, grid, field.data)

    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    (all(field.data[1-Hx:0,   1:Ny,   1:Nz] .== field[Nx-Hx+1:Nx, 1:Ny,           1:Nz]) &&
     all(field.data[1:Nx,   1-Hy:0,   1:Nz] .== field[1:Nx,      Ny-Hy+1:Ny,      1:Nz]) &&
     all(field.data[1:Nx,     1:Ny, 1-Hz:0] .== field[1:Nx,      1:Ny,      Nz-Hz+1:Nz]))
end

function multiple_halo_regions_correctly_filled(arch, FT, Nx, Ny, Nz)
    # Just choose something anisotropic to catch Δx/Δy type errors.
    Lx, Ly, Lz = 100, 200, 300

    grid = RegularCartesianGrid(FT, (Nx, Ny, Nz), (Lx, Ly, Lz))
    field1 = CellField(FT, arch, grid)
    field2 = FaceFieldX(FT, arch, grid)

    data(field1) .= rand(FT, Nx, Ny, Nz)
    data(field2) .= rand(FT, Nx, Ny, Nz)
    fill_halo_regions!(arch, grid, field1.data, field2.data)

    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    (all(field1.data[1-Hx:0,   1:Ny,   1:Nz] .== field1[Nx-Hx+1:Nx, 1:Ny,           1:Nz]) &&
     all(field1.data[1:Nx,   1-Hy:0,   1:Nz] .== field1[1:Nx,      Ny-Hy+1:Ny,      1:Nz]) &&
     all(field1.data[1:Nx,     1:Ny, 1-Hz:0] .== field1[1:Nx,      1:Ny,      Nz-Hz+1:Nz]) &&
     all(field2.data[1-Hx:0,   1:Ny,   1:Nz] .== field2[Nx-Hx+1:Nx, 1:Ny,           1:Nz]) &&
     all(field2.data[1:Nx,   1-Hy:0,   1:Nz] .== field2[1:Nx,      Ny-Hy+1:Ny,      1:Nz]) &&
     all(field2.data[1:Nx,     1:Ny, 1-Hz:0] .== field2[1:Nx,      1:Ny,      Nz-Hz+1:Nz]))
end
