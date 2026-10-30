include(joinpath(@__DIR__, "..", "setup", "reactant_test_utils.jl"))
using Reactant: @trace

#####
##### Reactant FFT-based NonhydrostaticModel tests
#####
# Groups tests by model configuration to avoid rebuilding the same grid+model.
# Uses @trace with track_numbers=false for time-stepping (see B.6.5 in differentiability-mwe.mdc).
# Both QB2 and RK3 timesteppers are supported.

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
        # model = NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2)
        model = NonhydrostaticModel(grid; timestepper=:RungeKutta3)

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

    @testset "2D (Bounded, Bounded, Flat) matches the CPU" begin
        # Bounded directions use DCTs built from FFTs, as on GPUs
        models = map((reactant_arch, CPU())) do arch
            grid = RectilinearGrid(arch; size=(8, 8), x=(0, 1), y=(0, 1), topology=(Bounded, Bounded, Flat))
            NonhydrostaticModel(grid; timestepper=:RungeKutta3, advection=Centered(), closure=ScalarDiffusivity(ν=1e-3))
        end
        reactant_model, cpu_model = models
        u₀ = [sin(π * x) * cos(π * y / 2) for x in range(0, 1, length=9), y in range(1/16, 15/16, length=8), z in 1:1]
        v₀ = [cos(π * x / 2) * sin(π * y) for x in range(1/16, 15/16, length=8), y in range(0, 1, length=9), z in 1:1]
        Δt = 0.01
        Nt = 4

        # set! projects the velocities with the pressure solver, which must run compiled
        function run_from!(model, u₀, v₀, Δt, Nt)
            set!(model, u=u₀, v=v₀)
            @trace track_numbers=false for i in 1:Nt
                time_step!(model, Δt)
            end
            return nothing
        end

        u₀r = Reactant.to_rarray(u₀)
        v₀r = Reactant.to_rarray(v₀)
        compiled_run! = @compile run_from!(reactant_model, u₀r, v₀r, Δt, Nt)
        compiled_run!(reactant_model, u₀r, v₀r, Δt, Nt)

        set!(cpu_model, u=u₀, v=v₀)
        for _ in 1:Nt
            time_step!(cpu_model, Δt)
        end
        for name in (:u, :v)
            reactant_values = on_architecture(CPU(), interior(getproperty(reactant_model.velocities, name)))
            cpu_values = interior(getproperty(cpu_model.velocities, name))
            @test reactant_values ≈ cpu_values rtol=1e-10
        end
    end
end
