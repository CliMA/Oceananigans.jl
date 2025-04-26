using Pkg

include("dependencies_for_runtests.jl")

group = get(ENV, "TEST_GROUP", "all") |> Symbol
test_file = get(ENV, "TEST_FILE", :none) |> Symbol

# if we are testing just a single file then group = :none
# to skip the full test suite
if test_file != :none
    group = :none
end

#####
##### Run tests
#####

CUDA.allowscalar() do

@testset "Oceananigans" begin

    if test_file != :none
        @testset "Single file test" begin
            include(String(test_file))
        end
    end

    # Initialization steps
    if group == :init || group == :all
        include("test_init.jl")
    end

    # Core Oceananigans
    if group == :unit || group == :all
        @testset "Unit tests" begin
            include("test_grids.jl")
            include("test_immersed_boundary_grid.jl")
            include("test_operators.jl")
            include("test_vector_rotation_operators.jl")
            include("test_boundary_conditions.jl")
            include("test_field.jl")
            include("test_regrid.jl")
            include("test_field_scans.jl")
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

    if group == :tripolar_grid || group == :all
        @testset "TripolarGrid tests" begin
            include("test_tripolar_grid.jl")
        end
    end

    if group == :poisson_solvers_1 || group == :all
        @testset "Poisson Solvers 1" begin
            include("test_poisson_solvers.jl")
        end
    end

    if group == :poisson_solvers_2 || group == :all
        @testset "Poisson Solvers 2" begin
            include("test_poisson_solvers_stretched_grids.jl")
            include("test_conjugate_gradient_poisson_solver.jl")
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
            include("test_krylov_solver.jl")
        end
    end

    # Simulations
    if group == :simulation || group == :all
        @testset "Simulation tests" begin
            include("test_simulations.jl")
            include("test_diagnostics.jl")
            include("test_output_writers.jl")
            include("test_netcdf_writer.jl")
            include("test_output_readers.jl")
        end
    end

    # Lagrangian particle tracking
    if group == :lagrangian_particles || group == :all
        @testset "Lagrangian particle tracking tests" begin
            include("test_lagrangian_particle_tracking.jl")
        end
    end

    # Models
    if group == :time_stepping_1 || group == :all
        @testset "Model and time stepping tests (part 1)" begin
            include("test_nonhydrostatic_models.jl")
            include("test_time_stepping.jl")
            include("test_active_cells_map.jl")
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
            include("test_biogeochemistry.jl")
            include("test_seawater_density.jl")
            include("test_orthogonal_spherical_shell_time_stepping.jl")
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
            include("test_multi_region_cubed_sphere.jl")
        end
    end

    if group == :distributed || group == :all
        MPI.Initialized() || MPI.Init()
        # In case CUDA is not found, we reset CUDA and restart the julia session
        reset_cuda_if_necessary()
        include("test_distributed_models.jl")
    end

    if group == :distributed_solvers || group == :all
        MPI.Initialized() || MPI.Init()
        # In case CUDA is not found, we reset CUDA and restart the julia session
        reset_cuda_if_necessary()
        include("test_distributed_transpose.jl")
        include("test_distributed_poisson_solvers.jl")
        include("test_distributed_macros.jl")
    end

    if group == :distributed_hydrostatic_model || group == :all
        MPI.Initialized() || MPI.Init()
        # In case CUDA is not found, we reset CUDA and restart the julia session
        reset_cuda_if_necessary()
        archs = test_architectures()
        include("test_hydrostatic_regression.jl")
        include("test_distributed_hydrostatic_model.jl")
    end

    # if group == :distributed_output || group == :all
    #     @testset "Distributed output writing and reading tests" begin
    #         include("test_distributed_output.jl")
    #     end
    # end

    if group == :distributed_nonhydrostatic_regression || group == :all
        MPI.Initialized() || MPI.Init()
        # In case CUDA is not found, we reset CUDA and restart the julia session
        reset_cuda_if_necessary()
        archs = nonhydrostatic_regression_test_architectures()
        include("test_nonhydrostatic_regression.jl")
    end

    if group == :nonhydrostatic_regression || group == :all
        include("test_nonhydrostatic_regression.jl")
    end

    if group == :hydrostatic_regression || group == :all
        include("test_hydrostatic_regression.jl")
    end

    if group == :scripts || group == :all
        @testset "Scripts" begin
            include("test_validation.jl")
        end
    end

    if group == :vertical_coordinate || group == :all
        @testset "Vertical coordinate tests" begin
            include("test_zstar_coordinate.jl")
        end
    end

    # Tests for MPI extension
    if group == :mpi_tripolar || group == :all
        @testset "Distributed tripolar tests" begin
            include("test_mpi_tripolar.jl")
        end
    end

    
    # Tests for Enzyme extension
    if group == :enzyme || group == :all
        @testset "Enzyme extension tests" begin
            include("test_enzyme.jl")
        end
    end

    # Tests for Reactant extension
    if group == :reactant_1 || group == :all
        @testset "Reactant extension tests 1" begin
            include("test_reactant.jl")
        end
    end

    if group == :reactant_2 || group == :all
        @testset "Reactant extension tests 2" begin
            include("test_reactant_latitude_longitude_grid.jl")
        end
    end

    if group == :sharding || group == :all
        @testset "Sharding Reactant extension tests" begin
            # Broken for the moment (trying to fix them in https://github.com/CliMA/Oceananigans.jl/pull/4293)
            # include("test_sharded_lat_lon.jl")
            # include("test_sharded_tripolar.jl")
        end
    end

    # Tests for Metal extension
    if group == :metal || group == :all
        @testset "Metal extension tests" begin
            include("test_metal.jl")
        end
    end

    # Tests for AMDGPU extension
    if group == :amdgpu || group == :all
        @testset "AMDGPU extension tests" begin
            include("test_amdgpu.jl")
        end
    end

    # Tests for oneAPI extension
    if group == :oneapi || group == :all
        @testset "oneAPI extension tests" begin
            include("test_oneapi.jl")
        end
    end

    if group == :convergence
        include("test_convergence.jl")
    end
end

end #CUDA.allowscalar()

