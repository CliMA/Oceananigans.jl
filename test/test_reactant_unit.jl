using Test
using Reactant
using CUDA
using Oceananigans
using Oceananigans.Architectures: ReactantState

arch = ReactantState()

@testset "Reactant unit tests" begin
    @testset "KernelFunctionOperation reductions on LatitudeLongitudeGrid" begin
        grid = LatitudeLongitudeGrid(arch;
                                     size = (40, 40, 10),
                                     longitude = (0, 360),
                                     latitude = (-10, 10),
                                     z = (-1000, 0),
                                     halo = (5, 5, 5))

        returnone(i, j, k, grid) = 1
        kfo = KernelFunctionOperation{Center, Center, Center}(returnone, grid)

        @test minimum(kfo) == 1
        @test maximum(kfo) == 1

        @test minimum_xspacing(grid) > 0
        @test minimum_yspacing(grid) > 0
        @test minimum_zspacing(grid) > 0

        @test minimum_xspacing(grid) ≈ minimum(xspacings(grid))
        @test minimum_yspacing(grid) ≈ minimum(yspacings(grid))
        @test minimum_zspacing(grid) ≈ minimum(zspacings(grid))
    end

    @testset "KernelFunctionOperation reductions on RectilinearGrid" begin
        grid = RectilinearGrid(arch;
                               size = (10, 10, 10),
                               extent = (1, 1, 1))

        returnone(i, j, k, grid) = 1
        kfo = KernelFunctionOperation{Center, Center, Center}(returnone, grid)

        @test minimum(kfo) == 1
        @test maximum(kfo) == 1

        @test minimum_xspacing(grid) > 0
        @test minimum_yspacing(grid) > 0
        @test minimum_zspacing(grid) > 0

        @test minimum_xspacing(grid) ≈ minimum(xspacings(grid))
        @test minimum_yspacing(grid) ≈ minimum(yspacings(grid))
        @test minimum_zspacing(grid) ≈ minimum(zspacings(grid))
    end

    @testset "Field reductions on LatitudeLongitudeGrid" begin
        grid = LatitudeLongitudeGrid(arch;
                                     size = (10, 10, 10),
                                     longitude = (0, 360),
                                     latitude = (-60, 60),
                                     z = (-1000, 0),
                                     halo = (5, 5, 5))

        c = CenterField(grid)
        set!(c, (x, y, z) -> z)

        @test minimum(c) ≈ -950.0
        @test maximum(c) ≈ -50.0
    end

    @testset "Field reductions on RectilinearGrid" begin
        grid = RectilinearGrid(arch;
                               size = (10, 10, 10),
                               extent = (1, 1, 1))

        c = CenterField(grid)
        set!(c, (x, y, z) -> z)

        @test minimum(c) ≈ -0.95
        @test maximum(c) ≈ -0.05
    end
end
