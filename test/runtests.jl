using Test

import FFTW

using Oceananigans
using Oceananigans.Operators

archs = [:CPU]
@hascuda archs = [:CPU, :GPU]

new_archs = [CPU()]
@hascuda new_archs = [CPU(), GPU()]

float_types = [Float32, Float64]

@testset "Oceananigans" begin
    println("Testing Oceananigans...")

    @testset "Grid" begin
        println("  Testing grids...")
        include("test_grids.jl")

        @testset "Grid initialization" begin
            println("    Testing grid initialization...")
            for ft in float_types
                @test correct_grid_size(ft)
                @test correct_cell_volume(ft)
                @test faces_start_at_zero(ft)
                @test end_faces_match_grid_length(ft)
            end
        end

        @testset "Grid dimensions" begin
            println("    Testing grid dimensions...")
            L = (100, 100, 100)
            for ft in float_types
                @test isbitstype(typeof(RegularCartesianGrid(ft, (16, 16, 16), (1, 1, 1))))

                @test RegularCartesianGrid(ft, (25, 25, 25), L).dim == 3
                @test RegularCartesianGrid(ft, (5, 25, 125), L).dim == 3
                @test RegularCartesianGrid(ft, (64, 64, 64), L).dim == 3
                @test RegularCartesianGrid(ft, (32, 32,  1), L).dim == 2
                @test RegularCartesianGrid(ft, (32,  1, 32), L).dim == 2
                @test RegularCartesianGrid(ft, (1,  32, 32), L).dim == 2
                @test RegularCartesianGrid(ft, (1,  1,  64), L).dim == 1

                @test_throws ArgumentError RegularCartesianGrid(ft, (32,), L)
                @test_throws ArgumentError RegularCartesianGrid(ft, (32, 64), L)
                @test_throws ArgumentError RegularCartesianGrid(ft, (1, 1, 1), L)
                @test_throws ArgumentError RegularCartesianGrid(ft, (32, 32, 32, 16), L)
                @test_throws ArgumentError RegularCartesianGrid(ft, (32, 32, 32), (100,))
                @test_throws ArgumentError RegularCartesianGrid(ft, (32, 32, 32), (100, 100))
                @test_throws ArgumentError RegularCartesianGrid(ft, (32, 32, 32), (100, 100, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(ft, (32, 32, 32), (100, 100, -100))

                @test_throws ArgumentError RegularCartesianGrid(ft, (32, 32, 32.0), (1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(ft, (20.1, 32, 32), (1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(ft, (32, nothing, 32), (1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(ft, (32, "32", 32), (1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(ft, (32, 32, 32), (1, nothing, 1))
                @test_throws ArgumentError RegularCartesianGrid(ft, (32, 32, 32), (1, "1", 1))
            end
        end
    end

    @testset "Fields" begin
        println("  Testing fields...")
        include("test_fields.jl")

        N = (4, 6, 8)
        L = (2π, 3π, 5π)

        field_types = [CellField, FaceFieldX, FaceFieldY, FaceFieldZ, EdgeField]

        @testset "Field initialization" begin
            println("    Testing field initialization...")
            for arch in new_archs, ft in float_types
                grid = RegularCartesianGrid(ft, N, L)

                for field_type in field_types
                    @test correct_field_size(arch, grid, field_type)
                end
            end
        end

        int_vals = Any[0, Int8(-1), Int16(2), Int32(-3), Int64(4), Int128(-5)]
        uint_vals = Any[6, UInt8(7), UInt16(8), UInt32(9), UInt64(10), UInt128(11)]
        float_vals = Any[0.0, -0.0, 6e-34, 1.0f10]
        rational_vals = Any[1//11, -23//7]
        other_vals = Any[π]
        vals = vcat(int_vals, uint_vals, float_vals, rational_vals, other_vals)

        @testset "Setting fields" begin
            println("    Testing field setting...")

            for arch in new_archs, ft in float_types
                grid = RegularCartesianGrid(ft, N, L)

                for field_type in field_types, val in vals
                    @test correct_field_value_was_set(arch, grid, field_type, val)
                end
            end
        end

        # @testset "Field operations" begin
        #     for arch in new_archs, ft in float_types
        #         grid = RegularCartesianGrid(ft, N, L)
        #
        #         for field_type in field_types, val1 in vals, val2 in vals
        #             @test correct_field_addition(arch, grid, field_type, val1, val2)
        #         end
        #     end
        # end
    end

    @testset "Operators" begin
        println("  Testing operators...")

        @testset "2D operators" begin
            println("    Testing 2D operators...")
            Nx, Ny, Nz = 32, 16, 8
            Lx, Ly, Lz = 100, 100, 100
            A3 = rand(Nx, Ny, Nz)

            test_indices_3d = [(4, 5, 5), (21, 11, 4), (16, 8, 4),  (30, 12, 3), (11, 3, 6), # Interior
                               (2, 10, 4), (31, 5, 6), (10, 2, 4), (17, 15, 5), (17, 10, 2), (23, 5, 7),  # Borderlands
                               (1, 5, 5), (32, 10, 3), (16, 1, 4), (16, 16, 4), (16, 8, 1), (16, 8, 8),  # Edges
                               (1, 1, 1), (32, 16, 8)] # Corners

            test_indices_2d_yz = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2),
                                  (1, 1, 5), (1, 5, 1), (1, 5, 5), (1, 11, 4),
                                  (1, 15, 7), (1, 15, 8), (1, 16, 7), (1, 16, 8)]

            test_indices_2d_xz = [(1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 1, 2),
                                  (1, 1, 5), (5, 1, 1), (5, 1, 5), (17, 1, 4),
                                  (31, 1, 7), (31, 1, 8), (32, 1, 7), (32, 1, 8)]

            A2yz = A3[1:1, :, :]  # A yz-slice with Nx==1.
            for idx in test_indices_2d_yz
                @test δx_f2c(A2yz, 1, idx...) ≈ 0
                @test δx_c2f(A2yz, 1, idx...) ≈ 0
                @test δy_f2c(A2yz, Ny, idx...) ≈ δy_f2c(A3, Ny, idx...)
                @test δy_c2f(A2yz, Ny, idx...) ≈ δy_c2f(A3, Ny, idx...)
                @test δz_f2c(A2yz, Nz, idx...) ≈ δz_f2c(A3, Nz, idx...)
                @test δz_c2f(A2yz, Nz, idx...) ≈ δz_c2f(A3, Nz, idx...)
            end

            A2xz = A3[:, 1:1, :]  # An xz-slice with Ny==1.
            for idx in test_indices_2d_xz
                @test δx_f2c(A2xz, Nx, idx...) ≈ δx_f2c(A3, Nx, idx...)
                @test δx_c2f(A2xz, Nx, idx...) ≈ δx_c2f(A3, Nx, idx...)
                @test δy_f2c(A2xz, 1, idx...) ≈ 0
                @test δy_c2f(A2xz, 1, idx...) ≈ 0
                @test δz_f2c(A2xz, Nz, idx...) ≈ δz_f2c(A3, Nz, idx...)
                @test δz_c2f(A2xz, Nz, idx...) ≈ δz_c2f(A3, Nz, idx...)
            end
        end
    end

    @testset "Poisson solvers" begin
        println("  Testing Poisson solvers...")
        include("test_poisson_solvers.jl")

        @testset "FFTW commutativity" begin
            println("    Testing FFTW commutativity...")

            # Testing for nice powers of two and not-so-nice prime numbers.
            for N in [4, 7, 8, 10, 48, 64, 79, 128]
                @test mixed_fft_commutes(N)
                @test mixed_ifft_commutes(N)
            end
        end

        @testset "FFTW plans" begin
            println("    Testing FFTW planning...")

            for ft in float_types
                @test fftw_planner_works(ft, 32, 32, 32, FFTW.ESTIMATE)
                @test fftw_planner_works(ft, 1,  32, 32, FFTW.ESTIMATE)
                @test fftw_planner_works(ft, 32,  1, 32, FFTW.ESTIMATE)
                @test fftw_planner_works(ft,  1,  1, 32, FFTW.ESTIMATE)
            end
        end

        @testset "Divergence-free solution [CPU]" begin
            println("    Testing divergence-free solution [CPU]...")

            for N in [5, 7, 10, 15, 20, 29, 32]
                for ft in float_types
                    @test poisson_ppn_planned_div_free_cpu(ft, N, N, N, FFTW.ESTIMATE)
                    @test poisson_ppn_planned_div_free_cpu(ft, 1, N, N, FFTW.ESTIMATE)
                    @test poisson_ppn_planned_div_free_cpu(ft, N, 1, N, FFTW.ESTIMATE)
                    @test poisson_ppn_planned_div_free_cpu(ft, 1, 1, N, FFTW.ESTIMATE)

                    # Commented because https://github.com/climate-machine/Oceananigans.jl/issues/99
                    # for planner_flag in [FFTW.ESTIMATE, FFTW.MEASURE]
                    #     @test test_3d_poisson_ppn_planned!_div_free(mm, N, N, N, planner_flag)
                    #     @test test_3d_poisson_ppn_planned!_div_free(mm, 1, N, N, planner_flag)
                    #     @test test_3d_poisson_ppn_planned!_div_free(mm, N, 1, N, planner_flag)
                    # end
                end
            end

            Ns = [5, 7, 10, 15, 20, 29, 32]
            for Nx in Ns, Ny in Ns, Nz in Ns, ft in float_types
                @test poisson_ppn_planned_div_free_cpu(ft, Nx, Ny, Nz, FFTW.ESTIMATE)
            end
        end
    end

    @testset "Model" begin
        for arch in archs, ft in float_types
            model = Model(N=(4, 5, 6), L=(1, 2, 3), arch=arch, float_type=ft)

            # Just testing that a Model was constructed with no errors.
            @test typeof(model) <: Model
        end
    end

    @testset "Boundary conditions" begin
        include("test_boundary_conditions.jl")

        Nx, Ny, Nz = 3, 4, 5 # for simple test
        funbc(args...) = π

        for fld in (:u, :v, :T, :S)
            for bctype in (Gradient, Flux)
                for bc in (0.6, rand(Nx, Ny), funbc)
                    @test test_z_boundary_condition_simple(fld, bctype, bc, Nx, Ny, Nz)
                end
            end
            @test test_diffusion_simple(fld)
            @test test_diffusion_budget(fld)
            @test test_diffusion_cosine(fld)
            @test test_flux_budget(fld)
        end
    end

    @testset "Forcing" begin
        add_one(args...) = 1.0
        function test_forcing(fld)
            kwarg = Dict(Symbol(:F, fld)=>add_one)
            forcing = Forcing(; kwarg...)
            f = getfield(forcing, fld)
            f() == 1.0
        end

        for fld in (:u, :v, :w, :T, :S)
            @test test_forcing(fld)
        end
    end

    @testset "Time stepping" begin
        include("test_time_stepping.jl")

        for arch in archs, ft in float_types
            @test test_basic_timestepping(arch, ft)
        end

        @testset "Adams-Bashforth 2" begin
            for arch in archs, ft in float_types
                run_first_AB2_time_step_tests(arch, ft)
            end
        end
    end

    @testset "Output writers" begin
        include("test_output_writers.jl")

        @testset "Checkpointing" begin
            run_thermal_bubble_checkpointer_tests()
        end

        @testset "NetCDF" begin
            run_thermal_bubble_netcdf_tests()
        end
    end

    @testset "Golden master tests" begin
        include("test_golden_master.jl")

        @testset "Thermal bubble" begin
            run_thermal_bubble_golden_master_tests()
        end

        @testset "Deep convection" begin
            run_deep_convection_golden_master_tests()
        end
    end
end # Oceananigans tests
