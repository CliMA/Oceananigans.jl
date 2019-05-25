using
    Test,
    Statistics,
    Oceananigans,
    Oceananigans.Operators,
    Oceananigans.TurbulenceClosures

import FFTW

import GPUifyLoops: @launch, @loop, @synchronize

archs = (CPU(),)
@hascuda archs = (CPU(), GPU())
@hascuda using CuArrays, CUDAnative

float_types = (Float32, Float64)

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
            for arch in archs, ft in float_types
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

            for arch in archs, ft in float_types
                grid = RegularCartesianGrid(ft, N, L)

                for field_type in field_types, val in vals
                    @test correct_field_value_was_set(arch, grid, field_type, val)
                end
            end
        end

        # @testset "Field operations" begin
        #     for arch in archs, ft in float_types
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

            grid = RegularCartesianGrid((Nx, Ny, Nz), (Lx, Ly, Lz))
            A3 = rand(Nx, Ny, Nz)

            A2yz = A3[1:1, :, :]  # A yz-slice with Nx==1.
            grid_yz = RegularCartesianGrid((1, Ny, Nz), (Lx, Ly, Lz))

            A2xz = A3[:, 1:1, :]  # An xz-slice with Ny==1.
            grid_xz = RegularCartesianGrid((Nx, 1, Nz), (Lx, Ly, Lz))

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

            for idx in test_indices_2d_yz
                @test δx_f2c(grid_yz, A2yz, idx...) ≈ 0
                @test δx_c2f(grid_yz, A2yz, idx...) ≈ 0
                @test δy_f2c(grid_yz, A2yz, idx...) ≈ δy_f2c(grid_yz, A3, idx...)
                @test δy_c2f(grid_yz, A2yz, idx...) ≈ δy_c2f(grid_yz, A3, idx...)
                @test δz_f2c(grid_yz, A2yz, idx...) ≈ δz_f2c(grid_yz, A3, idx...)
                @test δz_c2f(grid_yz, A2yz, idx...) ≈ δz_c2f(grid_yz, A3, idx...)
            end

            for idx in test_indices_2d_xz
                @test δx_f2c(grid_xz, A2xz, idx...) ≈ δx_f2c(grid_xz, A3, idx...)
                @test δx_c2f(grid_xz, A2xz, idx...) ≈ δx_c2f(grid_xz, A3, idx...)
                @test δy_f2c(grid_xz, A2xz, idx...) ≈ 0
                @test δy_c2f(grid_xz, A2xz, idx...) ≈ 0
                @test δz_f2c(grid_xz, A2xz, idx...) ≈ δz_f2c(grid_xz, A3, idx...)
                @test δz_c2f(grid_xz, A2xz, idx...) ≈ δz_c2f(grid_xz, A3, idx...)
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

            for T in float_types
                for arch in archs
                    @test fftw_planner_works(T, 32, 32, 32, FFTW.ESTIMATE, arch)
                end
                @test fftw_planner_works(T, 1,  32, 32, FFTW.ESTIMATE)
                @test fftw_planner_works(T, 32,  1, 32, FFTW.ESTIMATE)
                @test fftw_planner_works(T,  1,  1, 32, FFTW.ESTIMATE)
            end
        end

        @testset "Divergence-free solution" begin
            println("    Testing divergence-free solution...")

            for T in float_types
                for N in (32, 64)
                    for arch in archs
                        @test poisson_ppn_planned_div_free(T, N, N, N, FFTW.ESTIMATE, arch)
                    end

                    @test poisson_ppn_planned_div_free(T, 1, N, N, FFTW.ESTIMATE, CPU())
                    @test poisson_ppn_planned_div_free(T, N, 1, N, FFTW.ESTIMATE, CPU())
                    @test poisson_ppn_planned_div_free(T, 1, 1, N, FFTW.ESTIMATE, CPU())

                    # Commented because https://github.com/climate-machine/Oceananigans.jl/issues/99
                    # for planner_flag in [FFTW.ESTIMATE, FFTW.MEASURE]
                    #     @test test_3d_poisson_ppn_planned!_div_free(mm, N, N, N, planner_flag)
                    #     @test test_3d_poisson_ppn_planned!_div_free(mm, 1, N, N, planner_flag)
                    #     @test test_3d_poisson_ppn_planned!_div_free(mm, N, 1, N, planner_flag)
                    # end
                end
            end
        end

        @testset "Analytic solution reconstruction" begin
            println("    Testing analytic solution reconstruction...")
            for N in [32, 48, 64], m in [1, 2, 3]
                @test poisson_ppn_recover_sine_cosine_solution(Float64, N, N, N, 100, 100, 100, m, m, m)
            end
        end
    end

    @testset "Model" begin
        println("  Testing model...")

        for arch in archs, ft in float_types
            model = Model(N=(4, 5, 6), L=(1, 2, 3), arch=arch, float_type=ft)

            # Just testing that a Model was constructed with no errors/crashes.
            @test true
        end
    end

    @testset "Time stepping" begin
        println("  Testing time stepping...")
        include("test_time_stepping.jl")

        for arch in archs, ft in float_types
            @test time_stepping_works(arch, ft)
        end

        @testset "Adams-Bashforth 2" begin
            for arch in archs, ft in float_types
                run_first_AB2_time_step_tests(arch, ft)
            end
        end

        @testset "Recomputing w from continuity" begin
            for ft in float_types
                test_compute_w_from_continuity(CPU(), ft)
            end
        end

        @testset "Incompressibility" begin
            for ft in float_types, Nt in [1, 10, 100]
                @test incompressible_in_time(CPU(), ft, Nt)
            end
        end
    end

    @testset "Boundary conditions" begin
        println("  Testing boundary conditions...")
        include("test_boundary_conditions.jl")

        funbc(args...) = π

        Nx = Ny = 16
        for arch in archs
            for TF in float_types
                for fld in (:u, :v, :T, :S)
                    for bctype in (Gradient, Flux, Value)

                        arraybc = rand(TF, Nx, Ny)
                        if arch == GPU()
                            arraybc = CuArray(arraybc)
                        end

                        for bc in (TF(0.6), arraybc, funbc)
                            @test test_z_boundary_condition_simple(arch, TF, fld, bctype, bc, Nx, Ny)
                        end
                    end
                    @test test_z_boundary_condition_top_bottom_alias(arch, TF, fld)
                    @test test_z_boundary_condition_array(arch, TF, fld)
                    @test test_flux_budget(arch, TF, fld)
                end
            end
        end
    end

    @testset "Forcing" begin
        println("  Testing forcings...")
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

    @testset "Output writers" begin
        println("  Testing output writers...")
        include("test_output_writers.jl")

        @testset "Checkpointing" begin
            println("    Testing checkpointing...")
            run_thermal_bubble_checkpointer_tests()
        end

        @testset "NetCDF" begin
            println("    Testing NetCDF output writer...")
            run_thermal_bubble_netcdf_tests()
        end
    end

    @testset "Golden master tests" begin
        include("test_golden_master.jl")

        @testset "Thermal bubble" begin
            for arch in archs
                run_thermal_bubble_golden_master_tests(arch)
            end
        end

        @testset "Deep convection" begin
            run_deep_convection_golden_master_tests()
        end
    end

    @testset "Dynamics tests" begin
        println("  Testing dynamics...")
        include("test_dynamics.jl")
        @test internal_wave_test()
        @test passive_tracer_advection_test()

        for fld in (:u, :v, :T, :S)
            @test test_diffusion_simple(fld)
            @test test_diffusion_budget(fld)
            @test test_diffusion_cosine(fld)
        end
    end

    @testset "Turbulence closures tests" begin
        println("  Testing turbulence closures...")
        include("test_turbulence_closures.jl")
        @test test_function_interpolation()
        @test test_function_differentiation()

        for T in float_types
            for closure in (:ConstantIsotropicDiffusivity, :ConstantAnisotropicDiffusivity,
                            :ConstantSmagorinsky)
                @test test_closure_instantiation(T, closure)
            end

            @test test_constant_isotropic_diffusivity_basic(T)
            @test test_constant_isotropic_diffusivity_fluxdiv(T)
            @test test_anisotropic_diffusivity_fluxdiv(T, νv=zero(T), νh=zero(T))
            @test test_anisotropic_diffusivity_fluxdiv(T)

            @test test_smag_divflux_finiteness(T)
        end
    end
end # Oceananigans tests
