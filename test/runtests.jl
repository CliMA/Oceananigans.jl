using Test

using Oceananigans
using Oceananigans.Operators

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

    g32 = RegularCartesianGrid(N, L, Float32)  # Float32 RegularCartesianGrid
    g64 = RegularCartesianGrid(N, L, Float64)  # Float64 RegularCartesianGrid

    int_vals = Any[0, Int8(-1), Int16(2), Int32(-3), Int64(4), Int128(-5)]
    uint_vals = Any[6, UInt8(7), UInt16(8), UInt32(9), UInt64(10), UInt128(11)]
    vals = vcat(int_vals, uint_vals)

    # TODO: Use ≈ for floating-point values and set! should correctly convert
    # Rational and Irrational to Float32.
    # float_vals = Any[0.0, -0.0, 6e-34, 1f10]
    # rational_vals = Any[1//11, -22//7]
    # other_vals = Any[π]
    # vals = vcat(int_vals, uint_vals, float_vals, rational_vals, other_vals)

    for g in [g32, g64]
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

@testset "Operators" begin
    include("test_operators.jl")

    N = (20, 20, 20)
    L = (1000, 1000, 1000)

    g32 = RegularCartesianGrid(N, L, Float32)  # Float32 RegularCartesianGrid
    g64 = RegularCartesianGrid(N, L, Float64)  # Float64 RegularCartesianGrid

    for g in [g32, g64]
        fC = CellField(g)
        ffX = FaceFieldX(g)
        ffY = FaceFieldY(g)
        ffZ = FaceFieldZ(g)

        @test test_δxc2f(g)
        @test test_δxf2c(g)
        @test test_δyc2f(g)
        @test test_δyf2c(g)
        @test test_δzc2f(g)
        @test test_δzf2c(g)

        @test test_avgxc2f(g)
        @test test_avgxf2c(g)
        @test test_avgyc2f(g)
        @test test_avgyf2c(g)
        @test test_avgzc2f(g)
        @test test_avgzf2c(g)

        @test test_divf2c(g)
        @test test_divc2f(g)
        @test test_div_flux(g)

        @test test_u_dot_grad_u(g)
        @test test_u_dot_grad_v(g)
        @test test_u_dot_grad_w(g)

        for f in (fC, ffX, ffY, ffZ)
            # Fields should be initialized to zero.
            @test f.data ≈ zeros(size(f))

            # Calling with the wrong signature, e.g. two CellFields should error.
            for δ in (δx!, δy!, δz!)
                @test_throws MethodError δ(g, f, f)
            end
            for avg in (avgx!, avgy!, avgz!)
                @test_throws MethodError avg(g, f, f)
            end
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
