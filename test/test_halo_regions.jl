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
