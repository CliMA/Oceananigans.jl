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
    @test test_solve_poisson_1d_pbc_cosine_source()
    @test test_solve_poisson_1d_pbc_cosine_source_multiple_resolutions()
    for N in [10, 50, 100, 500, 1000, 10000]
        @test test_solve_poisson_1d_pbc_divergence_free(N)
    end
    @test test_solve_poisson_2d_pbc_gaussian_source()
    @test test_solve_poisson_2d_pbc_gaussian_source_multiple_resolutions()
    @test test_solve_poisson_2d_pbc_gaussian_source_Nx_eq_2Ny()
    @test test_solve_poisson_2d_pbc_gaussian_source_Ny_eq_2Nx()
    @test test_solve_poisson_2d_pbc_gaussian_source_Nx_eq_2Ny_multiple_resolutions()
    for N in [10, 50, 100, 500, 1000, 2000]
        @test test_solve_poisson_2d_pbc_divergence_free(N)
    end
    @test test_mixed_fft_commutativity()
    @test test_mixed_ifft_commutativity()
end
