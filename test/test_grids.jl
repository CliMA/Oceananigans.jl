function correct_grid_size(FT)
    grid = RegularCartesianGrid(FT; size=(4, 6, 8), length=(2π, 4π, 9π))

    # Checking ≈ as the grid could be storing Float32 values.
    return (grid.Nx ≈ 4  && grid.Ny ≈ 6  && grid.Nz ≈ 8 &&
            grid.Lx ≈ 2π && grid.Ly ≈ 4π && grid.Lz ≈ 9π)
end

function faces_start_at_zero(FT)
    grid = RegularCartesianGrid(FT; size=(10, 10, 10), length=(2π, 2π, 2π))
    return grid.xF[1] == 0 && grid.yF[1] == 0 && grid.zF[end] == 0
end

function end_faces_match_grid_length(FT)
    grid = RegularCartesianGrid(FT; size=(12, 13, 14), length=(π, π^2, π^3))
    return (grid.xF[end] - grid.xF[1] ≈ π   &&
            grid.yF[end] - grid.yF[1] ≈ π^2 &&
            grid.zF[end] - grid.zF[1] ≈ π^3)
end

function ranges_have_correct_length(FT)
    Nx, Ny, Nz = 8, 9, 10
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(1, 1, 1))
    return (length(grid.xC) == Nx && length(grid.xF) == Nx+1 &&
            length(grid.yC) == Ny && length(grid.yF) == Ny+1 &&
            length(grid.zC) == Nz && length(grid.zF) == Nz+1)
end

# See: https://github.com/climate-machine/Oceananigans.jl/issues/480
function no_roundoff_error_in_ranges(FT)
    Nx, Ny, Nz = 1, 1, 64
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(1, 1, π/2))
    return length(grid.zC) == Nz
end

@testset "Grids" begin
    println("Testing grids...")

    @testset "Grid initialization" begin
        println("  Testing grid initialization...")
        for FT in float_types
            @test correct_grid_size(FT)
            @test faces_start_at_zero(FT)
            @test end_faces_match_grid_length(FT)
            @test ranges_have_correct_length(FT)
            @test no_roundoff_error_in_ranges(FT)
        end
    end

    @testset "Grid dimensions" begin
        println("  Testing grid constructor errors...")
        L = (100, 100, 100)
        for FT in float_types
            @test isbitstype(typeof(RegularCartesianGrid(FT; size=(16, 16, 16), length=(1, 1, 1))))

            @test_throws ArgumentError RegularCartesianGrid(FT; size=(32,), length=L)
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, 64), length=L)
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, 32, 32, 16), length=L)

            @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, 32, 32.0), length=(1, 1, 1))
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(20.1, 32, 32), length=(1, 1, 1))
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, nothing, 32), length=(1, 1, 1))
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, "32", 32), length=(1, 1, 1))
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, 32, 32), length=(1, nothing, 1))
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, 32, 32), length=(1, "1", 1))

            @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16))
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), x=2)
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), y=[1, 2])
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), z=(-π, π))
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), x=1, y=2, z=3)
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), x=(0, 1), y=(0, 2), z=4)
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), x=(-1//2, 1), y=(1//7, 5//7), z=("0", "1"))
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), x=(-1//2, 1), y=(1//7, 5//7), z=(1, 2, 3))
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), x=(1, 0), y=(1//7, 5//7), z=(1, 2))
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), x=(0, 1), y=(1, 5), z=(π, -π))
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), x=(0, 1), y=(1, 5), z=(π, -π))
            @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), length=(1, 2, 3), x=(0, 1), y=(1, 5), z=(-π, π))
        end
    end
end
