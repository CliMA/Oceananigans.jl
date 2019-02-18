using Test

import FFTW

using Oceananigans
using Oceananigans.Operators

@testset "Oceananigans" begin

    @testset "Grid" begin
        include("test_grids.jl")

        @testset "Grid initialization" begin
            for arch in [:cpu], ft in [Float64]
                mm = ModelMetadata(arch, ft)
                @test test_grid_size(mm)
                @test test_cell_volume(mm)
                @test test_faces_start_at_zero(mm)
            end
        end

        @testset "Grid dimensions" begin
            L = (100, 100, 100)
            for arch in [:cpu], ft in [Float64, Float32, Float16]
                mm = ModelMetadata(arch, ft)
                @test RegularCartesianGrid(mm, (25, 25, 25), L).dim == 3
                @test RegularCartesianGrid(mm, (5, 25, 125), L).dim == 3
                @test RegularCartesianGrid(mm, (64, 64, 64), L).dim == 3
                @test RegularCartesianGrid(mm, (32, 32,  1), L).dim == 2
                @test RegularCartesianGrid(mm, (32,  1, 32), L).dim == 2
                @test RegularCartesianGrid(mm, (1,  32, 32), L).dim == 2
                @test_throws AssertionError RegularCartesianGrid(mm, (32,), L)
                @test_throws AssertionError RegularCartesianGrid(mm, (32, 64), L)
                @test_throws AssertionError RegularCartesianGrid(mm, (1, 1, 1), L)
                @test_throws AssertionError RegularCartesianGrid(mm, (32, 32, 32, 16), L)
                @test_throws AssertionError RegularCartesianGrid(mm, (32, 32, 32), (100,))
                @test_throws AssertionError RegularCartesianGrid(mm, (32, 32, 32), (100, 100))
                @test_throws AssertionError RegularCartesianGrid(mm, (32, 32, 32), (100, 100, 1, 1))
                @test_throws AssertionError RegularCartesianGrid(mm, (32, 32, 32), (100, 100, -100))
            end
        end
    end

    @testset "Fields" begin
        include("test_fields.jl")

        N = (4, 6, 8)
        L = (2π, 3π, 5π)

        int_vals = Any[0, Int8(-1), Int16(2), Int32(-3), Int64(4), Int128(-5)]
        uint_vals = Any[6, UInt8(7), UInt16(8), UInt32(9), UInt64(10), UInt128(11)]
        vals = vcat(int_vals, uint_vals)

        # TODO: Use ≈ for floating-point values and set! should correctly convert
        # Rational and Irrational to Float32.
        # float_vals = Any[0.0, -0.0, 6e-34, 1f10]
        # rational_vals = Any[1//11, -22//7]
        # other_vals = Any[π]
        # vals = vcat(int_vals, uint_vals, float_vals, rational_vals, other_vals)

        for arch in [:cpu], ft in [Float32, Float64]
            mm = ModelMetadata(arch, ft)
            grid = RegularCartesianGrid(mm, N, L)

            for field_type in [CellField, FaceFieldX, FaceFieldY, FaceFieldZ]
                @test test_init_field(mm, grid, field_type)

                for val in vals
                    @test test_set_field(mm, grid, field_type, val) || "type(g)=$(typeof(g)), ftf=$ftf, val=$val"
                end

                # TODO: Try adding together a bunch of different data types?
                @test test_add_field(mm, grid, field_type, 4, 6)
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

            domain_sizes = [(1000, 1000, 1000)]

            for N in grid_sizes, L in domain_sizes, arch in [:cpu], ft in [Float64, Float32]
                mm = ModelMetadata(arch, ft)
                g = RegularCartesianGrid(mm, N, L)

                @test test_δxc2f(mm, g)
                @test test_δxf2c(mm, g)
                @test test_δyc2f(mm, g)
                @test test_δyf2c(mm, g)
                @test test_δzc2f(mm, g)
                @test test_δzf2c(mm, g)

                @test test_avgxc2f(mm, g)
                @test test_avgxf2c(mm, g)
                @test test_avgyc2f(mm, g)
                @test test_avgyf2c(mm, g)
                @test test_avgzc2f(mm, g)
                @test test_avgzf2c(mm, g)

                @test test_divf2c(mm, g)
                @test test_divc2f(mm, g)
                @test test_div_flux(mm, g)

                @test test_u_dot_grad_u(mm, g)
                @test test_u_dot_grad_v(mm, g)
                @test test_u_dot_grad_w(mm, g) || "N=$(N), eltype(g)=$(eltype(g))"

                @test test_κ∇²(mm, g)
                @test test_𝜈∇²u(mm, g)
                @test test_𝜈∇²v(mm, g)
                @test test_𝜈∇²w(mm, g)

                fC = CellField(mm, g)
                ffX = FaceFieldX(mm, g)
                ffY = FaceFieldY(mm, g)
                ffZ = FaceFieldZ(mm, g)

                for f in [fC, ffX, ffY, ffZ]
                    # Fields should be initialized to zero.
                    @test f.data ≈ zeros(size(f))

                    # Calling with the wrong signature, e.g. two CellFields should error.
                    for δ in [δx!, δy!, δz!]
                        @test_throws MethodError δ(g, f, f)
                    end
                    for avg in [avgx!, avgy!, avgz!]
                        @test_throws MethodError avg(g, f, f)
                    end
                end
            end
        end

        @testset "Laplacian" begin
            N = (20, 20, 20)
            L = (20, 20, 20)

            for arch in [:cpu], ft in [Float64, Float32]
                mm = ModelMetadata(arch, ft)
                g = RegularCartesianGrid(mm, N, L)
                @test test_∇²_ppn(mm, g)
            end
        end
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

            for arch in [:cpu], ft in [Float64]
                mm = ModelMetadata(arch, ft)

                @test test_3d_poisson_solver_ppn!_div_free(mm, N, N, N)
                @test test_3d_poisson_solver_ppn!_div_free(mm, 1, N, N)
                @test test_3d_poisson_solver_ppn!_div_free(mm, N, 1, N)

                for planner_flag in [FFTW.ESTIMATE, FFTW.MEASURE]
                    @test test_3d_poisson_ppn_planned!_div_free(mm, N, N, N, FFTW.ESTIMATE)
                    @test test_3d_poisson_ppn_planned!_div_free(mm, 1, N, N, FFTW.ESTIMATE)
                    @test test_3d_poisson_ppn_planned!_div_free(mm, N, 1, N, FFTW.ESTIMATE)
                end
            end
        end

        for Nx in [5, 10, 20, 50, 100], Ny in [5, 10, 20, 50, 100], Nz in [10, 20, 50]
            @test test_3d_poisson_solver_ppn_div_free(Nx, Ny, Nz)

            for arch in [:cpu], ft in [Float64]
                mm = ModelMetadata(arch, ft)
                @test test_3d_poisson_solver_ppn!_div_free(mm, Nx, Ny, Nz)

                for planner_flag in [FFTW.ESTIMATE, FFTW.MEASURE]
                    @test test_3d_poisson_ppn_planned!_div_free(mm, Nx, Ny, Nz, FFTW.ESTIMATE)
                end
            end
        end

        for planner_flag in [FFTW.ESTIMATE, FFTW.MEASURE], arch in [:cpu], ft in [Float64]
            mm = ModelMetadata(arch, ft)
            @test test_fftw_planner(mm, 100, 100, 100, FFTW.ESTIMATE)
            @test test_fftw_planner(mm, 1, 100, 100, FFTW.ESTIMATE)
            @test test_fftw_planner(mm, 100, 1, 100, FFTW.ESTIMATE)
        end
    end

    @testset "Model" begin
        model = Model((32, 32, 16), (2000, 2000, 1000))
        @test typeof(model) == Model  # Just testing that no errors happen.
    end

    @testset "Time stepping" begin
        Nx, Ny, Nz = 100, 1, 50
        Lx, Ly, Lz = 2000, 1, 1000
        Nt, Δt = 10, 20
        ΔR = 10

        model = Model((Nx, Ny, Nz), (Lx, Ly, Lz))
        time_step!(model; Nt=Nt, Δt=Δt)

        @test typeof(model) == Model  # Just testing that no errors happen.
    end

    @testset "Elementwise operator kernels" begin
        include("../src/operators/ops_regular_cartesian_grid_elementwise.jl")

        Nx, Ny, Nz = 32, 16, 8
        Lx, Ly, Lz = 100, 100, 100

        model = Model((Nx, Ny, Nz), (Lx, Ly, Lz))
        g, stmp, otmp = model.grid, model.stepper_tmp, model.operator_tmp
        U, tr = model.velocities, model.tracers

        test_indices = [(4, 5, 5), (21, 11, 4), (16, 8, 4),  # Interior
                        (17, 10, 2), (23, 5, 7),  # Borderlands
                        (1, 5, 5), (32, 10, 3), (16, 1, 4), (16, 16, 4), (16, 8, 1), (16, 8, 8),  # Edges
                        (1, 1, 1), (32, 16, 8)]  # Corners

        f, δxf = stmp.fC1, stmp.fFX
        @. f.data = rand()
        Oceananigans.Operators.δx!(g, f, δxf)
        for idx in test_indices; @test δx_c2f(g, f, idx...) ≈ δxf.data[idx...]; end
        for idx in test_indices; @test δx_c2f(f, g.Nx, idx...) ≈ δxf.data[idx...]; end

        f, δyf = stmp.fC1, stmp.fFY
        @. f.data = rand()
        Oceananigans.Operators.δy!(g, f, δyf)
        for idx in test_indices; @test δy_c2f(g, f, idx...) ≈ δyf.data[idx...]; end

        f, δzf = stmp.fC1, stmp.fFZ
        @. f.data = rand()
        Oceananigans.Operators.δz!(g, f, δzf)
        for idx in test_indices; @test δz_c2f(g, f, idx...) ≈ δzf.data[idx...]; end

        u, v, w, div_u = U.u, U.v, U.w, stmp.fC1
        @. u.data = rand(); @. v.data = rand(); @. w.data = rand();
        Oceananigans.Operators.div!(g, u, v, w, div_u, otmp)
        for idx in test_indices; @test div(g, u, v, w, idx...) ≈ div_u.data[idx...]; end

        u, T, uT̄ˣ, δx_uT̄ˣ = U.u, tr.T, stmp.fFX, stmp.fC1
        @. u.data = rand(); @. T.data = rand();
        Oceananigans.Operators.avgx!(g, T, uT̄ˣ)
        @. uT̄ˣ.data = u.data * uT̄ˣ.data
        Oceananigans.Operators.δx!(g, uT̄ˣ, δx_uT̄ˣ)
        for idx in test_indices; @test δx_f2c_ab̄ˣ(g, u, T, idx...) ≈ δx_uT̄ˣ.data[idx...]; end

        v, T, vT̄ʸ, δy_vT̄ʸ = U.v, tr.T, stmp.fFY, stmp.fC1
        @. v.data = rand(); @. T.data = rand();
        Oceananigans.Operators.avgy!(g, T, vT̄ʸ)
        @. vT̄ʸ.data = v.data * vT̄ʸ.data
        Oceananigans.Operators.δy!(g, vT̄ʸ, δy_vT̄ʸ)
        for idx in test_indices; @test δy_f2c_ab̄ʸ(g, v, T, idx...) ≈ δy_vT̄ʸ.data[idx...]; end

        w, T, wT̄ᶻ, δz_wT̄ᶻ = U.w, tr.T, stmp.fFZ, stmp.fC1
        @. w.data = rand(); @. T.data = rand();
        Oceananigans.Operators.avgz!(g, T, wT̄ᶻ)
        @. wT̄ᶻ.data = w.data * wT̄ᶻ.data
        Oceananigans.Operators.δz!(g, wT̄ᶻ, δz_wT̄ᶻ)
        for idx in test_indices; @test δz_f2c_ab̄ᶻ(g, w, T, idx...) ≈ δz_wT̄ᶻ.data[idx...]; end

        u, v, w, T, div_uT = U.u, U.v, U.w, tr.T, stmp.fC1
        @. u.data = rand(); @. v.data = rand(); @. w.data = rand(); @. T.data = rand();
        Oceananigans.Operators.div_flux!(g, u, v, w, T, div_uT, otmp)
        for idx in test_indices; @test div_flux(g, U, T, idx...) ≈ div_uT.data[idx...]; end

        u, u̅ˣ, ∂uu = U.u, stmp.fC1, stmp.fFX
        @. u.data = rand();
        Oceananigans.Operators.avgx!(g, u, u̅ˣ)
        @. u̅ˣ.data = u̅ˣ.data^2
        Oceananigans.Operators.δx!(g, u̅ˣ, ∂uu)
        for idx in test_indices; @test δx_c2f_ūˣūˣ(g, u, idx...) ≈ ∂uu.data[idx...]; end

        u, v, w, u_grad_u = U.u, U.v, U.w, stmp.fFX
        @. u.data = rand(); @. v.data = rand(); @. w.data = rand();
        Oceananigans.Operators.u∇u!(g, U, u_grad_u, otmp)
        for idx in test_indices; @test u∇u(g, U, idx...) ≈ u_grad_u.data[idx...]; end

        u, v, w, u_grad_v = U.u, U.v, U.w, stmp.fFY
        @. u.data = rand(); @. v.data = rand(); @. w.data = rand();
        Oceananigans.Operators.u∇v!(g, U, u_grad_v, otmp)
        for idx in test_indices; @test u∇v(g, U, idx...) ≈ u_grad_v.data[idx...]; end

        u, w, w̅ˣ, u̅ᶻ, ∂wu = U.u, U.w, otmp.fE1, otmp.fE2, stmp.fFZ
        Oceananigans.Operators.avgx!(g, w, w̅ˣ)
        Oceananigans.Operators.avgz!(g, u, u̅ᶻ)
        wu = otmp.fE1
        @. wu.data = w̅ˣ.data * u̅ᶻ.data
        Oceananigans.Operators.δx!(g, wu, ∂wu)
        for idx in test_indices; @test δx_e2f_ūᶻw̄ˣ(g, u, w, idx...) ≈ ∂wu.data[idx...]; end

        v, w, v̄ᶻ, w̅ʸ, ∂wv = U.v, U.w, otmp.fE1, otmp.fE2, stmp.fFZ
        Oceananigans.Operators.avgz!(g, v, v̄ᶻ)
        Oceananigans.Operators.avgy!(g, w, w̅ʸ)
        wv = otmp.fE1
        @. wv.data = v̄ᶻ.data * w̅ʸ.data
        Oceananigans.Operators.δy!(g, wv, ∂wv)
        for idx in test_indices; @test δy_e2f_v̄ᶻw̄ʸ(g, v, w, idx...) ≈ ∂wv.data[idx...]; end

        w, w̄ᶻ, ∂ww = U.w, stmp.fC1, stmp.fFZ
        @. w.data = rand();
        Oceananigans.Operators.avgz!(g, w, w̄ᶻ)
        @. w̄ᶻ.data = w̄ᶻ.data^2
        Oceananigans.Operators.δz!(g, w̄ᶻ, ∂ww)
        for idx in test_indices; @test δz_c2f_w̄ᶻw̄ᶻ(g, w, idx...) ≈ ∂ww.data[idx...]; end

        u, v, w, u_grad_w = U.u, U.v, U.w, stmp.fFZ
        @. u.data = rand(); @. v.data = rand(); @. w.data = rand();
        Oceananigans.Operators.u∇w!(g, U, u_grad_w, otmp)
        for idx in test_indices; @test u∇w(g, U, idx...) ≈ u_grad_w.data[idx...]; end

        T, δxT, δx²T = tr.T, stmp.fFX, stmp.fC1
        @. T.data = rand();
        Oceananigans.Operators.δx!(g, T, δxT)
        Oceananigans.Operators.δx!(g, δxT, δx²T)
        for idx in test_indices; @test δx²_c2f2c(g, T, idx...) ≈ δx²T.data[idx...]; end

        T, δyT, δy²T = tr.T, stmp.fFY, stmp.fC1
        @. T.data = rand();
        Oceananigans.Operators.δy!(g, T, δyT)
        Oceananigans.Operators.δy!(g, δyT, δy²T)
        for idx in test_indices; @test δy²_c2f2c(g, T, idx...) ≈ δy²T.data[idx...]; end

        T, δzT, δz²T = tr.T, stmp.fFZ, stmp.fC1
        @. T.data = rand();
        Oceananigans.Operators.δz!(g, T, δzT)
        Oceananigans.Operators.δz!(g, δzT, δz²T)
        for idx in test_indices; @test δz²_c2f2c(g, T, idx...) ≈ δz²T.data[idx...]; end

        κh, κv = 4e-2, 4e-2
        T, κ∇²T = tr.T, stmp.fC1
        @. T.data = rand();
        Oceananigans.Operators.κ∇²!(g, T, κ∇²T, κh, κv, otmp)
        for idx in test_indices; @test κ∇²(g, T, κh, κv, idx...) ≈ κ∇²T.data[idx...]; end

        𝜈h, 𝜈v = 4e-2, 4e-2
        u, 𝜈_lap_u = U.u, stmp.fFX
        @. u.data = rand();
        Oceananigans.Operators.𝜈∇²u!(g, u, 𝜈_lap_u, 𝜈h, 𝜈v, otmp)
        for idx in test_indices; @test 𝜈∇²u(g, u, 𝜈h, 𝜈v, idx...) ≈ 𝜈_lap_u.data[idx...]; end

        𝜈h, 𝜈v = 4e-2, 4e-2
        v, 𝜈_lap_v = U.v, stmp.fFY
        @. v.data = rand();
        Oceananigans.Operators.𝜈∇²v!(g, v, 𝜈_lap_v, 𝜈h, 𝜈v, otmp)
        for idx in test_indices; @test 𝜈∇²v(g, v, 𝜈h, 𝜈v, idx...) ≈ 𝜈_lap_v.data[idx...]; end

        𝜈h, 𝜈v = 4e-2, 4e-2
        v, 𝜈_lap_w = U.w, stmp.fFZ
        @. w.data = rand();
        Oceananigans.Operators.𝜈∇²w!(g, w, 𝜈_lap_w, 𝜈h, 𝜈v, otmp)
        for idx in test_indices; @test 𝜈∇²w(g, w, 𝜈h, 𝜈v, idx...) ≈ 𝜈_lap_w.data[idx...]; end
    end
end
