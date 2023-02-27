include("dependencies_for_runtests.jl")

CUDA.allowscalar() do

@testset "Oceananigans" begin
    if test_file != :none
        @testset "Single file test" begin
            include(String(test_file))
        end
    end

    # Core Oceananigans 
    if group == :unit || group == :all
        @testset "Unit tests" begin
            include("test_grids.jl")
            include("test_operators.jl")
            include("test_boundary_conditions.jl")
            include("test_field.jl")
            include("test_field_reductions.jl")
            include("test_halo_regions.jl")
            include("test_coriolis.jl")
            include("test_buoyancy.jl")
            include("test_stokes_drift.jl")
            include("test_utils.jl")
            include("test_schedules.jl")
        end
    end

    if group == :abstract_operations || group == :all
        @testset "AbstractOperations and broadcasting tests" begin
            include("test_abstract_operations.jl")
            include("test_conditional_reductions.jl")
            include("test_computed_field.jl")
            include("test_broadcasting.jl")
        end
    end

    if group == :poisson_solvers_1 || group == :all
        @testset "Poisson Solvers 1" begin
            include("test_poisson_solvers.jl")
        end
    end

    if group == :poisson_solvers_2 || group == :all
        @testset "Poisson Solvers 2" begin
            include("test_poisson_solvers_vertically_stretched_grid.jl")
        end
    end

    if group == :matrix_poisson_solvers || group == :all
        @testset "Matrix Poisson Solvers" begin
            include("test_matrix_poisson_solver.jl")
        end
    end

    if group == :general_solvers || group == :all
        @testset "General Solvers" begin
            include("test_batched_tridiagonal_solver.jl")
            include("test_preconditioned_conjugate_gradient_solver.jl")
            include("test_multigrid_solver.jl")
        end
    end

    # Simulations
    if group == :simulation || group == :all
        @testset "Simulation tests" begin
            include("test_simulations.jl")
            include("test_diagnostics.jl")
            include("test_output_writers.jl")
            include("test_output_readers.jl")
        end
    end

    # Lagrangian particle tracking
    if group == :lagrangian || group == :all
        @testset "Lagrangian particle tracking tests" begin
            include("test_lagrangian_particle_tracking.jl")
        end
    end

    # Models
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
            include("test_immersed_advection.jl")
        end
    end

    if group == :time_stepping_3 || group == :all
        @testset "Model and time stepping tests (part 3)" begin
            include("test_dynamics.jl")
        end
    end

    if group == :turbulence_closures || group == :all
        @testset "Turbulence closures tests" begin
            include("test_turbulence_closures.jl")
        end
    end

    if group == :shallow_water || group == :all
        include("test_shallow_water_models.jl")
    end

    if group == :hydrostatic_free_surface || group == :all
        @testset "HydrostaticFreeSurfaceModel tests" begin
            include("test_hydrostatic_free_surface_models.jl")
            include("test_ensemble_hydrostatic_free_surface_models.jl")
            include("test_hydrostatic_free_surface_immersed_boundaries.jl")
            include("test_vertical_vorticity_field.jl")
            include("test_implicit_free_surface_solver.jl")
            include("test_split_explicit_free_surface_solver.jl")
            include("test_split_explicit_vertical_integrals.jl")
            include("test_hydrostatic_free_surface_immersed_boundaries_implicit_solve.jl")
        end
    end
    
    # Model enhancements: cubed sphere, distributed, etc
    if group == :multi_region || group == :all
        @testset "Multi Region tests" begin
            include("test_multi_region_unit.jl")
            include("test_multi_region_advection_diffusion.jl")
            include("test_multi_region_implicit_solver.jl")
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
    end

    if group == :distributed_solvers || group == :all
        MPI.Initialized() || MPI.Init()
        include("test_distributed_poisson_solvers.jl")
    end

    if group == :nonhydrostatic_regression || group == :all
        include("test_nonhydrostatic_regression.jl")
    end

    if group == :hydrostatic_regression || group == :all
        include("test_hydrostatic_regression.jl")
    end

    if group == :shallowwater_regression || group == :all
        include("test_shallow_water_regression.jl")
    end

    if group == :scripts || group == :all
        @testset "Scripts" begin
            include("test_validation.jl")
        end
    end

    if group == :convergence
        include("test_convergence.jl")
    end

    if group == :quick_amd
        include("test_quick_amd.jl")
    end
end

end #CUDA.allowscalar()