include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: zspacings
using Oceananigans.ImmersedBoundaries: PartialCellBottom

@testset "PartialCellBottom fraction validation" begin
    @testset "Invalid fractions [$FT]" for FT in float_types
        for fraction in (FT(-1), FT(2), FT(NaN), FT(Inf), FT(-Inf), -eps(FT), nextfloat(one(FT)))
            @test_throws ArgumentError PartialCellBottom(zero(FT); minimum_fractional_cell_height=fraction)
        end
    end

    @testset "Non-real fractions" begin
        for fraction in (nothing, [1//2], 1//2 + im, "0.5")
            @test_throws ArgumentError PartialCellBottom(0; minimum_fractional_cell_height=fraction)
        end
    end

    @testset "Valid endpoints and types" begin
        for fraction in (0, 1, 1//2, Float32(1//2), Float64(1//2))
            bottom = PartialCellBottom(0; minimum_fractional_cell_height=fraction)
            @test bottom.minimum_fractional_cell_height === fraction
        end
    end

    @testset "Grid precision [$FT, $(typeof(arch))]" for arch in archs, FT in float_types
        underlying_grid = RectilinearGrid(arch, FT; size=4, z=(-1, 0), topology=(Flat, Flat, Bounded))

        for fraction in (0, 1//4, 1//2, 1)
            bottom = PartialCellBottom(-9//16; minimum_fractional_cell_height=fraction)
            grid = ImmersedBoundaryGrid(underlying_grid, bottom)
            materialized_fraction = grid.immersed_boundary.minimum_fractional_cell_height
            @test materialized_fraction isa FT
            @test materialized_fraction == FT(fraction)
        end

        bottom = PartialCellBottom(-9//16; minimum_fractional_cell_height=1//2)
        grid = ImmersedBoundaryGrid(underlying_grid, bottom)
        numerical_bottom = Array(interior(bottom_height_field(grid)))
        center_spacings = Array(interior(Field(zspacings(grid, Center(), Center(), Center()))))
        face_spacings = Array(interior(Field(zspacings(grid, Center(), Center(), Face()))))

        @test all(==(FT(-5//8)), numerical_bottom)
        @test vec(center_spacings) ≈ FT[1//4, 1//8, 1//4, 1//4]
        @test vec(face_spacings) ≈ FT[1//4, 1//4, 3//16, 1//4, 1//4]
    end

    @testset "Positive fractions must survive grid conversion [$(typeof(arch))]" for arch in archs
        underlying_grid = RectilinearGrid(arch, Float32; size=4, z=(-1, 0), topology=(Flat, Flat, Bounded))
        bottom = PartialCellBottom(-9//16; minimum_fractional_cell_height=nextfloat(zero(Float64)))
        @test_throws ArgumentError ImmersedBoundaryGrid(underlying_grid, bottom)
    end
end
