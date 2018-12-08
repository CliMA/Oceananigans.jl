using Test
using Oceananigans

# function testeos()
#   T0 = 283
#   S0 = 35
#   p0 = 1e5
#   rho0 = 1.027e3
#   ρ(T0, S0, p0) == rho0
# end
#
# @testset "Equation of State" begin
#   @test testeos()
# end

@testset "Grid" begin
    include("test_grid.jl")
    @test test_grid_size()

    d, n = 0.1, 4
    @test test_Δx(d, n)
    @test test_Δy(d, n)
    @test test_Δz(d, n)

    @test test_cell_volume()
    @test test_faces_start_at_zero()
end

@testset "Fields" begin
    include("test_field.jl")

    N = (4, 6, 8)
    L = (2π, 3π, 5π)

    gdf = RegularCartesianGrid(N, L)           # Default RegularCartesianGrid
    g32 = RegularCartesianGrid(N, L, Float32)  # Float32 RegularCartesianGrid
    g64 = RegularCartesianGrid(N, L, Float64)  # Float64 RegularCartesianGrid

    int_vals = Any[0, Int8(-1), Int16(2), Int32(-3), Int64(4), Int128(-5)]
    uint_vals = Any[6, UInt8(7), UInt16(8), UInt32(9), UInt64(10), UInt128(11)]
    float_vals = Any[0.0, -0.0, 6e-34, 1f10, π]
    rational_vals = Any[1//11, -22//7]
    vals = vcat(int_vals, uint_vals, float_vals, rational_vals)

    for g in [gdf, g32, g64]
        for ftf in (CellField, FaceFieldX, FaceFieldY, FaceFieldZ)
            @test test_init_field(g, ftf)

            for val in vals
                @test test_set_field(g, ftf, val) || "type(g)=$(typeof(g)), ftf=$ftf, val=$val"
            end

            # TODO: Try adding together a bunch of different data types.
            @test test_add_field(g, ftf, 4, 6)
        end
    end
end

    # @testset "Spectral solvers" begin
    #     include("test_spectral_solvers.jl")
    #     for N in [4, 8, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
    #         @test test_solve_poisson_1d_pbc_cosine_source(N)
    #     end
    #
    #     for N in [10, 50, 100, 500, 1000, 10000]
    #         @test test_solve_poisson_1d_pbc_divergence_free(N)
    #     end
    #
    #     for N in [32, 64, 128, 256, 512, 1024]
    #         @test test_solve_poisson_2d_pbc_gaussian_source(N, N)
    #         @test test_solve_poisson_2d_pbc_gaussian_source(2*N, N)
    #         @test test_solve_poisson_2d_pbc_gaussian_source(N, 2*N)
    #     end
    #
    #     for N in [10, 50, 100, 500, 1000, 2000]
    #         @test test_solve_poisson_2d_pbc_divergence_free(N)
    #     end
    #
    #     for N in [4, 8, 10, 64, 100, 256]
    #         @test test_mixed_fft_commutativity(N)
    #         @test test_mixed_ifft_commutativity(N)
    #     end
    #
    #     for N in [5, 10, 20, 50, 100]
    #         @test test_3d_poisson_solver_ppn_div_free(N, N, N)
    #     end
    # end
