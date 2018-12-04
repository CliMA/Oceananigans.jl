using
    OceanDispatch,
    Test

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
    @test test_Δx()
    @test test_Δy()
    @test test_Δz()
    @test test_cell_volume()
end


@testset "Spectral solvers" begin
    include("test_spectral_solvers.jl")
    for N in [4, 8, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
        @test test_solve_poisson_1d_pbc_cosine_source(N)
    end

    for N in [10, 50, 100, 500, 1000, 10000]
        @test test_solve_poisson_1d_pbc_divergence_free(N)
    end

    for N in [32, 64, 128, 256, 512, 1024]
        @test test_solve_poisson_2d_pbc_gaussian_source(N, N)
        @test test_solve_poisson_2d_pbc_gaussian_source(2*N, N)
        @test test_solve_poisson_2d_pbc_gaussian_source(N, 2*N)
    end

    for N in [10, 50, 100, 500, 1000, 2000]
        @test test_solve_poisson_2d_pbc_divergence_free(N)
    end

    for N in [4, 8, 10, 64, 100, 256]
        @test test_mixed_fft_commutativity(N)
        @test test_mixed_ifft_commutativity(N)
    end

    for N in [5, 10, 20, 50, 100]
        @test test_3d_poisson_solver_ppn_div_free(N, N, N)
    end
end
