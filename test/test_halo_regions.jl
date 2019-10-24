function halo_regions_initalized_correctly(arch, FT, Nx, Ny, Nz)
    # Just choose something anisotropic to catch Δx/Δy type errors.
    Lx, Ly, Lz = 10, 20, 30

    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))
    field = CellField(FT, arch, grid)

    # Fill the interior with random numbers.
    interior(field) .= rand(FT, Nx, Ny, Nz)

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

    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))
    field = CellField(FT, arch, grid)
    fbcs = HorizontallyPeriodicBCs()

    interior(field) .= rand(FT, Nx, Ny, Nz)
    fill_halo_regions!(field.data, fbcs, arch, grid)

    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    (all(field.data[1-Hx:0,   1:Ny,   1:Nz] .== field.data[Nx-Hx+1:Nx, 1:Ny,       1:Nz]) &&
     all(field.data[1:Nx,   1-Hy:0,   1:Nz] .== field.data[1:Nx,      Ny-Hy+1:Ny,  1:Nz]) &&
     all(field.data[1:Nx,     1:Ny,     1-Hz:0] .== field.data[1:Nx,        1:Ny,   1:1]) &&
     all(field.data[1:Nx,     1:Ny, Nz+1:Nz+Hz] .== field.data[1:Nx,        1:Ny, Nz:Nz]))
end

@testset "Halo regions" begin
    println("Testing halo regions...")

    Ns = [(8, 8, 8), (8, 8, 4), (10, 7, 5),
          (1, 8, 8), (1, 9, 5),
          (8, 1, 8), (5, 1, 9),
          (8, 8, 1), (5, 9, 1),
          (1, 1, 8)]

    @testset "Initializing halo regions" begin
        println("  Testing initializing halo regions...")
        for arch in archs, FT in float_types, N in Ns
            @test halo_regions_initalized_correctly(arch, FT, N...)
        end
    end

    @testset "Filling halo regions" begin
        println("  Testing filling halo regions...")
        for arch in archs, FT in float_types, N in Ns
            @test halo_regions_correctly_filled(arch, FT, N...)
        end
    end
end
