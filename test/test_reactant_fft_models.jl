include("reactant_test_utils.jl")
using Reactant: @trace

#####
##### Reactant FFT-based NonhydrostaticModel tests
#####
# Groups tests by model configuration to avoid rebuilding the same grid+model.
# Uses @trace with track_numbers=false for time-stepping (see B.6.5 in differentiability-mwe.mdc).
# Must use QB2 timestepper (RK3 not supported).

@testset "Reactant FFT-based NonhydrostaticModel" begin
    @info "Performing Reactant FFT-based NonhydrostaticModel tests..."
    reactant_arch = ReactantState()

    # Wrapper function with @trace and track_numbers=false
    function run_timesteps!(model, Δt, Nt)
        @trace track_numbers=false for i in 1:Nt
            time_step!(model, Δt)
        end
        return nothing
    end

    @testset "3D (Periodic, Periodic, Periodic)" begin
        @info "  Testing NonhydrostaticModel 3D (Periodic, Periodic, Periodic)..."
        grid = RectilinearGrid(reactant_arch; size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))
        model = NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2)

        @testset "Construction" begin
            @test model isa NonhydrostaticModel
            @test model.grid.architecture isa ReactantState
        end

        @testset "Compiled time_step!" begin
            @info "    Compiling and running time_step!..."
            Δt = 0.001
            Nt = 4
            compiled_run! = @compile run_timesteps!(model, Δt, Nt)
            compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end
    end

    @testset "2D (Periodic, Periodic, Flat)" begin
        @info "  Testing NonhydrostaticModel 2D (Periodic, Periodic, Flat)..."
        grid = RectilinearGrid(reactant_arch; size=(4, 4), extent=(1, 1),
                               topology=(Periodic, Periodic, Flat))
        model = NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2)

        @testset "Construction" begin
            @test model isa NonhydrostaticModel
        end

        @testset "Compiled time_step!" begin
            @info "    Compiling and running time_step!..."
            Δt = 0.001
            Nt = 4
            compiled_run! = @compile run_timesteps!(model, Δt, Nt)
            compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end
    end
end
