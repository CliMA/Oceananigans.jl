function correct_grid_size(ft::DataType)
    g = RegularCartesianGrid(ft, (4, 6, 8), (2π, 4π, 9π))

    # Checking ≈ as the grid could be storing Float32 values.
    (g.Nx ≈ 4  && g.Ny ≈ 6  && g.Nz ≈ 8 &&
     g.Lx ≈ 2π && g.Ly ≈ 4π && g.Lz ≈ 9π)
end

function correct_cell_volume(ft::DataType)
    Nx, Ny, Nz = 19, 13, 7
    Δx, Δy, Δz = 0.1, 0.2, 0.3
    Lx, Ly, Lz = Nx*Δx, Ny*Δy, Nz*Δz
    g = RegularCartesianGrid(ft, (Nx, Ny, Nz), (Lx, Ly, Lz))

    # Checking ≈ as the grid could be storing Float32 values.
    g.V ≈ Δx*Δy*Δz
end

function faces_start_at_zero(ft::DataType)
    g = RegularCartesianGrid(ft, (10, 10, 10), (2π, 2π, 2π))
    g.xF[1] == 0 && g.yF[1] == 0 && g.zF[end] == 0
end

function end_faces_match_grid_length(ft::DataType)
    g = RegularCartesianGrid(ft, (12, 13, 14), (π, π^2, π^3))
    (g.xF[end] - g.xF[1] ≈ π   &&
     g.yF[end] - g.yF[1] ≈ π^2 &&
     g.zF[end] - g.zF[1] ≈ π^3)
end

function ranges_have_correct_length(FT)
    Nx, Ny, Nz = 8, 9, 10
    grid = RegularCartesianGrid(FT, (Nx, Ny, Nz), (1, 1, 1))
    return (length(grid.xC) == Nx && length(grid.xF) == Nx+1 &&
            length(grid.yC) == Ny && length(grid.yF) == Ny+1 &&
            length(grid.zC) == Nz && length(grid.zF) == Nz+1)
end

@testset "Grids" begin
    println("Testing grids...")

    @testset "Grid initialization" begin
        println("  Testing grid initialization...")
        for FT in float_types
            @test correct_grid_size(FT)
            @test correct_cell_volume(FT)
            @test faces_start_at_zero(FT)
            @test end_faces_match_grid_length(FT)
            @test ranges_have_correct_length(FT)
        end
    end

    @testset "Grid dimensions" begin
        println("  Testing grid dimensions...")
        L = (100, 100, 100)
        for FT in float_types
            @test isbitstype(typeof(RegularCartesianGrid(FT, (16, 16, 16), (1, 1, 1))))

            @test_throws ArgumentError RegularCartesianGrid(FT, (32,), L)
            @test_throws ArgumentError RegularCartesianGrid(FT, (32, 64), L)
            @test_throws ArgumentError RegularCartesianGrid(FT, (32, 32, 32, 16), L)
            @test_throws ArgumentError RegularCartesianGrid(FT, (32, 32, 32), (100,))
            @test_throws ArgumentError RegularCartesianGrid(FT, (32, 32, 32), (100, 100))
            @test_throws ArgumentError RegularCartesianGrid(FT, (32, 32, 32), (100, 100, 1, 1))
            @test_throws ArgumentError RegularCartesianGrid(FT, (32, 32, 32), (100, 100, -100))

            @test_throws ArgumentError RegularCartesianGrid(FT, (32, 32, 32.0), (1, 1, 1))
            @test_throws ArgumentError RegularCartesianGrid(FT, (20.1, 32, 32), (1, 1, 1))
            @test_throws ArgumentError RegularCartesianGrid(FT, (32, nothing, 32), (1, 1, 1))
            @test_throws ArgumentError RegularCartesianGrid(FT, (32, "32", 32), (1, 1, 1))
            @test_throws ArgumentError RegularCartesianGrid(FT, (32, 32, 32), (1, nothing, 1))
            @test_throws ArgumentError RegularCartesianGrid(FT, (32, 32, 32), (1, "1", 1))
        end
    end
end
