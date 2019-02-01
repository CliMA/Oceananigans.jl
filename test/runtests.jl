using Test

import FFTW

using Oceananigans
using Oceananigans.Operators

@testset "Oceananigans" begin

    @testset "Grid" begin
        include("test_grids.jl")

        @testset "Grid initialization" begin
            @test test_grid_size()
            @test test_cell_volume()
            @test test_faces_start_at_zero()
        end

        @testset "Grid dimensions" begin
            L = (100, 100, 100)
            for ft in [Float32, Float64]
                @test RegularCartesianGrid((25, 25, 25), L; FloatType=ft).dim == 3
                @test RegularCartesianGrid((5, 25, 125), L; FloatType=ft).dim == 3
                @test RegularCartesianGrid((64, 64, 64), L; FloatType=ft).dim == 3
                @test RegularCartesianGrid((32, 32,  1), L; FloatType=ft).dim == 2
                @test RegularCartesianGrid((32,  1, 32), L; FloatType=ft).dim == 2
                @test RegularCartesianGrid((1,  32, 32), L; FloatType=ft).dim == 2
                @test_throws AssertionError RegularCartesianGrid((32,), L; FloatType=ft)
                @test_throws AssertionError RegularCartesianGrid((32, 64), L; FloatType=ft)
                @test_throws AssertionError RegularCartesianGrid((1, 64, 1), L; FloatType=ft)
                @test_throws AssertionError RegularCartesianGrid((1, 1, 1), L; FloatType=ft)
                @test_throws AssertionError RegularCartesianGrid((32, 32, 32, 16), L; FloatType=ft)
                @test_throws AssertionError RegularCartesianGrid((32, 32, 32), (100,); FloatType=ft)
                @test_throws AssertionError RegularCartesianGrid((32, 32, 32), (100, 100); FloatType=ft)
                @test_throws AssertionError RegularCartesianGrid((32, 32, 32), (100, 100, 1, 1); FloatType=ft)
                @test_throws AssertionError RegularCartesianGrid((32, 32, 32), (100, 100, -100); FloatType=ft)
            end
        end
    end

    @testset "Fields" begin
        include("test_fields.jl")

        N = (4, 6, 8)
        L = (2π, 3π, 5π)

        g32 = RegularCartesianGrid(N, L; FloatType=Float32)
        g64 = RegularCartesianGrid(N, L; FloatType=Float64)

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

        @testset "2D operators" begin
            Nx, Ny, Nz = 10, 10, 10
            A3 = rand(Nx, Ny, Nz)
            A2y = A3[:, 1:1, :]
            A2x = A3[1:1, :, :]

            @test δˣf2c(A2x) ≈ zeros(1, Ny, Nz)
            @test δˣc2f(A2x) ≈ zeros(1, Ny, Nz)
            @test δʸf2c(A2x) ≈ δʸf2c(A3)[1:1, :, :]
            @test δʸc2f(A2x) ≈ δʸc2f(A3)[1:1, :, :]
            @test δᶻf2c(A2x) ≈ δᶻf2c(A3)[1:1, :, :]
            @test δᶻc2f(A2x) ≈ δᶻc2f(A3)[1:1, :, :]

            @test δˣf2c(A2y) ≈ δˣf2c(A3)[:, 1:1, :]
            @test δˣc2f(A2y) ≈ δˣc2f(A3)[:, 1:1, :]
            @test δʸf2c(A2y) ≈ zeros(Nx, 1, Nz)
            @test δʸc2f(A2y) ≈ zeros(Nx, 1, Nz)
            @test δᶻf2c(A2y) ≈ δᶻf2c(A3)[:, 1:1, :]
            @test δᶻc2f(A2y) ≈ δᶻc2f(A3)[:, 1:1, :]
        end

        @testset "3D operators" begin
            grid_sizes = [(25, 25, 25), (64, 64, 64),
                          (16, 32, 32), (32, 16, 32), (16, 32, 32),
                          (1,  32, 32), (1, 16, 32),
                          (32,  1, 32), (32, 1, 16),
                          (32, 32,  1), (32, 16, 1)]

            L = (1000, 1000, 1000)

            for N in grid_sizes
                g32 = RegularCartesianGrid(N, L; FloatType=Float32)
                g64 = RegularCartesianGrid(N, L; FloatType=Float64)

                for g in [g32, g64]
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
                    @test test_u_dot_grad_w(g) || "N=$(N), eltype(g)=$(eltype(g))"

                    @test test_κ∇²(g)
                    @test test_𝜈∇²u(g)
                    @test test_𝜈∇²v(g)
                    @test test_𝜈∇²w(g)

                    fC = CellField(g)
                    ffX = FaceFieldX(g)
                    ffY = FaceFieldY(g)
                    ffZ = FaceFieldZ(g)

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
        end

        N = (20, 20, 20)
        L = (20, 20, 20)

        g32 = RegularCartesianGrid(N, L; FloatType=Float32)
        g64 = RegularCartesianGrid(N, L; FloatType=Float64)

        @test test_∇²_ppn(g32)
        @test test_∇²_ppn(g64)
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
            @test test_3d_poisson_solver_ppn_div_free(1, N, N)
            @test test_3d_poisson_solver_ppn_div_free(N, 1, N)
            @test test_3d_poisson_solver_ppn!_div_free(N, N, N)
            @test test_3d_poisson_solver_ppn!_div_free(1, N, N)
            @test test_3d_poisson_solver_ppn!_div_free(N, 1, N)

            for planner_flag in [FFTW.ESTIMATE, FFTW.MEASURE]
                @test test_3d_poisson_ppn_planned!_div_free(N, N, N, FFTW.ESTIMATE)
                @test test_3d_poisson_ppn_planned!_div_free(1, N, N, FFTW.ESTIMATE)
                @test test_3d_poisson_ppn_planned!_div_free(N, 1, N, FFTW.ESTIMATE)
            end
        end
        for Nx in [5, 10, 20, 50, 100], Ny in [5, 10, 20, 50, 100], Nz in [10, 20, 50]
            @test test_3d_poisson_solver_ppn_div_free(Nx, Ny, Nz)
            @test test_3d_poisson_solver_ppn!_div_free(Nx, Ny, Nz)

            for planner_flag in [FFTW.ESTIMATE, FFTW.MEASURE]
                @test test_3d_poisson_ppn_planned!_div_free(Nx, Ny, Nz, FFTW.ESTIMATE)
            end
        end

        for planner_flag in [FFTW.ESTIMATE, FFTW.MEASURE]
            @test test_fftw_planner(100, 100, 100, FFTW.ESTIMATE)
            @test test_fftw_planner(1, 100, 100, FFTW.ESTIMATE)
            @test test_fftw_planner(100, 1, 100, FFTW.ESTIMATE)
        end
    end

    @testset "Problem" begin
        p = Problem((32, 32, 16), (2000, 2000, 1000))
        @test typeof(p) == Problem  # Just testing that no errors happen.
    end

    @testset "Time stepping" begin
        Nx, Ny, Nz = 100, 1, 50
        Lx, Ly, Lz = 2000, 1, 1000
        Nt, Δt = 2500, 20
        ΔR = 10

        problem = Problem((Nx, Ny, Nz), (Lx, Ly, Lz))

        R  = SavedFields(problem.g, Nt, ΔR)

        time_stepping!(problem; Nt=Nt, Δt=Δt, R=R)

        @test typeof(problem) == Problem   # Just testing that no errors happen.
    end
end
