function correct_grid_size_regular(FT)
    grid = RegularCartesianGrid(FT; size=(4, 6, 8), length=(2π, 4π, 9π))

    # Checking ≈ as the grid could be storing Float32 values.
    return (grid.Nx ≈ 4  && grid.Ny ≈ 6  && grid.Nz ≈ 8 &&
            grid.Lx ≈ 2π && grid.Ly ≈ 4π && grid.Lz ≈ 9π)
end

function correct_halo_size_regular(FT)
    grid = RegularCartesianGrid(FT; size=(4, 6, 8), length=(2π, 4π, 9π), halo=(1, 2, 3))
    return (grid.Hx == 1  && grid.Hy == 2  && grid.Hz == 3)
end

function faces_start_at_zero_regular(FT)
    grid = RegularCartesianGrid(FT; size=(10, 10, 10), length=(2π, 2π, 2π))
    return grid.xF[1] == 0 && grid.yF[1] == 0 && grid.zF[end] == 0
end

function end_faces_match_grid_length_regular(FT)
    grid = RegularCartesianGrid(FT; size=(12, 13, 14), length=(π, π^2, π^3))
    return (grid.xF[end] - grid.xF[1] ≈ π   &&
            grid.yF[end] - grid.yF[1] ≈ π^2 &&
            grid.zF[end] - grid.zF[1] ≈ π^3)
end

function ranges_have_correct_length_regular(FT)
    Nx, Ny, Nz = 8, 9, 10
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(1, 1, 1))
    return (length(grid.xC) == Nx && length(grid.xF) == Nx+1 &&
            length(grid.yC) == Ny && length(grid.yF) == Ny+1 &&
            length(grid.zC) == Nz && length(grid.zF) == Nz+1)
end

# See: https://github.com/climate-machine/Oceananigans.jl/issues/480
function no_roundoff_error_in_ranges_regular(FT)
    Nx, Ny, Nz = 1, 1, 64
    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(1, 1, π/2))
    return length(grid.zC) == Nz && length(grid.zF) == Nz+1
end

function grid_properties_are_same_type_regular(FT)
    grid = RegularCartesianGrid(FT; size=(10, 10, 10), length=(1, 1//7, 2π))
    return all(isa.([grid.Lx, grid.Ly, grid.Lz, grid.Δx, grid.Δy, grid.Δz], FT)) &&
           all(eltype.([grid.xF, grid.yF, grid.zF, grid.xC, grid.yC, grid.zC]) .== FT)
end

function correct_constant_grid_spacings(FT)
    grid = VerticallyStretchedCartesianGrid(FT; size=(16, 16, 16), x=(0,1), y=(0,1), z=(0,16), zF=collect(0:16))
    return all(grid.ΔzF .== 1) && all(grid.ΔzC .== 1)
end

function correct_quadratic_grid_spacings(FT)
    Nx = Ny = Nz = 16
    grid = VerticallyStretchedCartesianGrid(FT; size=(Nx, Ny, Nz), x=(0,1), y=(0,1), z=(0,Nz^2), zF=collect(0:Nz).^2)

     zF(k) = (k-1)^2
     zC(k) = ((k-1)^2 + k^2) / 2
    ΔzF(k) = k^2 - (k-1)^2
    ΔzC(k) = 2k

     zF_is_correct = all(isapprox.(grid.zF,   zF.(1:Nz+1)))
     zC_is_correct = all(isapprox.(grid.zC,   zC.(1:Nz)))
    ΔzF_is_correct = all(isapprox.(grid.ΔzF, ΔzF.(1:Nz)))
    ΔzC_is_correct = all(isapprox.(grid.ΔzC, ΔzC.(1:Nz-1)))

    return zF_is_correct && zC_is_correct && ΔzF_is_correct && ΔzC_is_correct
end

function correct_tanh_grid_spacings(FT)
    Nx = Ny = Nz = 16

    S = 3  # Stretching factor
    zF(k) = tanh(S * (2*(k-1)/Nz - 1)) / tanh(S)

    grid = VerticallyStretchedCartesianGrid(FT; size=(Nx, Ny, Nz), x=(0,1), y=(0,1), z=(-1,1), zF=zF)

     zC(k) = (zF(k) + zF(k+1)) / 2
    ΔzF(k) = zF(k+1) - zF(k)
    ΔzC(k) = zC(k+1) - zC(k)

    zF_is_correct = all(isapprox.(grid.zF,   zF.(1:Nz+1)))
    zC_is_correct = all(isapprox.(grid.zC,   zC.(1:Nz)))
   ΔzF_is_correct = all(isapprox.(grid.ΔzF, ΔzF.(1:Nz)))
   ΔzC_is_correct = all(isapprox.(grid.ΔzC, ΔzC.(1:Nz-1)))

   return zF_is_correct && zC_is_correct && ΔzF_is_correct && ΔzC_is_correct
end

function grid_properties_are_same_type_stretched(FT)
    Nx, Ny, Nz = 16, 16, 16
    grid = VerticallyStretchedCartesianGrid(FT; size=(16, 16, 16), x=(0,1), y=(0,1), z=(0.0,16.0), zF=collect(0:16))
    return all(isa.([grid.Lx, grid.Ly, grid.Lz, grid.Δx, grid.Δy], FT)) &&
           all(eltype.([grid.ΔzF, grid.ΔzC, grid.xF, grid.yF, grid.zF, grid.xC, grid.yC, grid.zC]) .== FT)
end

@testset "Grids" begin
    println("Testing grids...")

    @testset "Regular Cartesian grid" begin
        println("  Testing regular Cartesian grid...")

        @testset "Grid initialization" begin
            println("    Testing grid initialization...")

            for FT in float_types
                @test correct_grid_size_regular(FT)
                @test correct_halo_size_regular(FT)
                @test faces_start_at_zero_regular(FT)
                @test end_faces_match_grid_length_regular(FT)
                @test ranges_have_correct_length_regular(FT)
                @test no_roundoff_error_in_ranges_regular(FT)
                @test grid_properties_are_same_type_regular(FT)
            end
        end

        @testset "Grid dimensions" begin
            println("    Testing grid constructor errors...")

            for FT in float_types
                @test isbitstype(typeof(RegularCartesianGrid(FT; size=(16, 16, 16), length=(1, 1, 1))))

                @test_throws ArgumentError RegularCartesianGrid(FT; size=(32,), length=(1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, 64), length=(1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, 32, 32, 16), length=(1, 1, 1))

                @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, 32, 32.0), length=(1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT; size=(20.1, 32, 32), length=(1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, nothing, 32), length=(1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, "32", 32), length=(1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, 32, 32), length=(1, nothing, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, 32, 32), length=(1, "1", 1))
                @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, 32, 32), length=(1, 1, 1), halo=(1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT; size=(32, 32, 32), length=(1, 1, 1), halo=(1.0, 1, 1))

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
                @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), length=(1, 2, 3), x=(0, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT; size=(16, 16, 16), length=(1, 2, 3), x=(0, 1), y=(1, 5), z=(-π, π))
            end
        end
    end

    @testset "Vertically stretched Cartesian grid" begin
        println("  Testing vertically stretched Cartesian grid...")

        @testset "Grid initialization" begin
            println("    Testing grid initialization...")

            for FT in float_types
                @test correct_constant_grid_spacings(FT)
                @test correct_quadratic_grid_spacings(FT)
                @test correct_tanh_grid_spacings(FT)
                @test grid_properties_are_same_type_stretched(FT)
            end
        end
    end
end
