include("reactant_test_utils.jl")
using Reactant: @trace

@testset "Reactant FFT-based model construction" begin
    @info "Performing Reactant FFT-based NonhydrostaticModel construction tests..."
    reactnt_arch = ReactantState()

    @testset "NonhydrostaticModel 3D (Periodic, Periodic, Periodic)" begin
        @info "  Testing NonhydrostaticModel 3D construction (Periodic, Periodic, Periodic)..."
        grid = RectilinearGrid(reactnt_arch; size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))

        model = NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2)
        @test model isa NonhydrostaticModel
        @test model.grid.architecture isa ReactantState
    end

    @testset "NonhydrostaticModel 2D (Periodic, Periodic, Flat)" begin
        @info "  Testing NonhydrostaticModel 2D construction (Periodic, Periodic, Flat)..."
        grid = RectilinearGrid(reactnt_arch; size=(4, 4), extent=(1, 1),
                               topology=(Periodic, Periodic, Flat))

        model = NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2)
        @test model isa NonhydrostaticModel
    end
end

#####
##### Time-stepping with @compile (compiled execution)
#####
# This follows the standard pattern used in Breeze.jl and differentiation tests:
# - Use @trace with track_numbers=false inside a wrapper function
# - Δt stays as Float64 (not traced), only arrays are traced
# - This avoids issues with Clock field assignments
# - Must use QB2 timestepper (RK3 not supported, see B.6.5)

@testset "Reactant FFT-based model time-stepping (compiled execution)" begin
    @info "Performing Reactant FFT-based NonhydrostaticModel time-stepping tests..."
    reactnt_arch = ReactantState()

    # Wrapper function with @trace and track_numbers=false
    function run_timesteps!(model, Δt, nsteps)
        @trace track_numbers=false for i in 1:nsteps
            time_step!(model, Δt)
        end
        return nothing
    end

    @testset "NonhydrostaticModel 3D compiled time_step!" begin
        @info "  Testing NonhydrostaticModel 3D compiled time_step! (with FFT pressure solver)..."
        grid = RectilinearGrid(reactnt_arch; size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))
        model = NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2)
        
        Δt = 0.001  # Regular Float64, not ConcreteRNumber
        nsteps = 1
        
        # Compile and execute
        compiled_run! = @compile run_timesteps!(model, Δt, nsteps)
        compiled_run!(model, Δt, nsteps)
        
        @test model.clock.iteration == 1
    end

    @testset "NonhydrostaticModel 2D compiled time_step!" begin
        @info "  Testing NonhydrostaticModel 2D compiled time_step! (with FFT pressure solver)..."
        grid = RectilinearGrid(reactnt_arch; size=(4, 4), extent=(1, 1),
                               topology=(Periodic, Periodic, Flat))
        model = NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2)
        
        Δt = 0.001
        nsteps = 1
        
        compiled_run! = @compile run_timesteps!(model, Δt, nsteps)
        compiled_run!(model, Δt, nsteps)
        
        @test model.clock.iteration == 1
    end
end
