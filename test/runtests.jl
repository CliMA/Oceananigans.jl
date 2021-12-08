
include("dependencies_for_runtests.jl")

@testset "Oceananigans" begin
    if group == :unit || group == :all
        @testset "Unit tests" begin
            include("test_grids.jl")
            include("test_operators.jl")
            include("test_boundary_conditions.jl")
            include("test_field.jl")
            include("test_reduced_field.jl")
            include("test_averaged_field.jl")
            include("test_kernel_computed_field.jl")
            include("test_halo_regions.jl")
            include("test_coriolis.jl")
            include("test_buoyancy.jl")
            include("test_stokes_drift.jl")
            include("test_utils.jl")
        end
    end

    if group == :solvers || group == :all
        @testset "Solvers" begin
            include("test_batched_tridiagonal_solver.jl")
            include("test_preconditioned_conjugate_gradient_solver.jl")
            include("test_poisson_solvers.jl")
            include("test_matrix_poisson_solver.jl")
        end
    end

    if group == :time_stepping_1 || group == :all
        @testset "Model and time stepping tests (part 1)" begin
            include("test_nonhydrostatic_models.jl")
            include("test_time_stepping.jl")
        end
    end

    if group == :time_stepping_2 || group == :all
        @testset "Model and time stepping tests (part 2)" begin
            include("test_boundary_conditions_integration.jl")
            include("test_forcings.jl")
            include("test_turbulence_closures.jl")
            include("test_dynamics.jl")
        end
    end

    if group == :shallow_water || group == :all
        include("test_shallow_water_models.jl")
    end

    if group == :hydrostatic_free_surface || group == :all
        @testset "HydrostaticFreeSurfaceModel tests" begin
            include("test_hydrostatic_free_surface_models.jl")
            include("test_hydrostatic_free_surface_immersed_boundaries.jl")
            include("test_vertical_vorticity_field.jl")
            include("test_implicit_free_surface_solver.jl")
            include("test_hydrostatic_free_surface_immersed_boundaries_apply_surf_bc.jl")
            include("test_hydrostatic_free_surface_immersed_boundaries_vertical_integrals.jl")
            include("test_hydrostatic_free_surface_immersed_boundaries_congrad_solve.jl")
        end
    end

    if group == :abstract_operations || group == :all
        @testset "AbstractOperations and broadcasting tests" begin
            include("test_abstract_operations.jl")
            include("test_computed_field.jl")
            include("test_broadcasting.jl")
        end
    end

    if group == :simulation || group == :all
        @testset "Simulation tests" begin
            include("test_simulations.jl")
            include("test_diagnostics.jl")
            include("test_output_writers.jl")
            include("test_output_readers.jl")
            include("test_lagrangian_particle_tracking.jl")
        end
    end

    if group == :cubed_sphere || group == :all
        @testset "Cubed sphere tests" begin
            include("test_cubed_spheres.jl")
            include("test_cubed_sphere_halo_exchange.jl")
            include("test_cubed_sphere_circulation.jl")
        end
    end

    if group == :distributed || group == :all
        MPI.Initialized() || MPI.Init()
        include("test_distributed_models.jl")
        include("test_distributed_poisson_solvers.jl")
    end

    if group == :regression || group == :all
        include("test_regression.jl")
    end

    if group == :scripts || group == :all
        @testset "Scripts" begin
            include("test_validation.jl")
        end
    end

    if group == :convergence
        include("test_convergence.jl")
    end
end
