include(joinpath(@__DIR__, "..", "setup", "reactant_test_utils.jl"))
using Reactant: @trace
using Enzyme
using Statistics: mean

#####
##### Reactant FFT-based NonhydrostaticModel tests
#####
# Groups tests by model configuration to avoid rebuilding the same grid+model.
# Uses @trace with track_numbers=false for time-stepping (see B.6.5 in differentiability-mwe.mdc).
# Both QB2 and RK3 timesteppers are supported.

@testset "Reactant FFT-based NonhydrostaticModel" begin
    arch = ReactantState()

    function run_timesteps!(model, Δt, Nt)
        @trace track_numbers=false for _ in 1:Nt
            time_step!(model, Δt)
        end
        return nothing
    end

    configs = [
        (topology = (Periodic, Periodic, Periodic), size = (16, 16, 16), extent = (1, 1, 1),
         timestepper = :RungeKutta3, name = "3D (Periodic, Periodic, Periodic) — RK3"),
        (topology = (Bounded, Bounded, Bounded), size = (16, 16, 16), extent = (1, 1, 1),
         timestepper = :QuasiAdamsBashforth2, name = "3D (Bounded, Bounded, Bounded) — QB2"),
    ]

    for config in configs
        @testset "$(config.name)" begin
            grid = RectilinearGrid(arch; size=config.size, extent=config.extent, topology=config.topology)
            model = NonhydrostaticModel(grid; timestepper=config.timestepper, tracers=:T)

            @testset "Construction" begin
                @test model isa NonhydrostaticModel
                @test model.grid.architecture isa ReactantState
            end

            @testset "Compiled time_step!" begin
                Δt = 0.001
                Nt = 4
                compiled_run! = @compile run_timesteps!(model, Δt, Nt)
                compiled_run!(model, Δt, Nt)
                @test model.clock.iteration == Nt
            end

            @testset "Enzyme reverse-mode gradient" begin
                dmodel = Enzyme.make_zero(model)

                T_init  = CenterField(grid)
                set!(T_init, (x, y, z) -> 0.01 * x + 0.01 * y)
                dT_init = CenterField(grid)
                set!(dT_init, 0.0)

                function loss(model, T_init, Δt, nsteps)
                    set!(model, T=T_init)
                    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
                        time_step!(model, Δt)
                    end
                    return mean(interior(model.tracers.T).^2)
                end

                function grad_loss(model, dmodel, T_init, dT_init, Δt, nsteps)
                    parent(dT_init) .= 0
                    _, loss_value = Enzyme.autodiff(
                        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                        loss, Enzyme.Active,
                        Enzyme.Duplicated(model, dmodel),
                        Enzyme.Duplicated(T_init, dT_init),
                        Enzyme.Const(Δt),
                        Enzyme.Const(nsteps))
                    return dT_init, loss_value
                end

                Δt     = 0.001
                nsteps = 4

                compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                    model, dmodel, T_init, dT_init, Δt, nsteps)
                @test compiled_grad !== nothing

                dT, loss_val = compiled_grad(model, dmodel, T_init, dT_init, Δt, nsteps)
                @test loss_val > 0
                @test !isnan(loss_val)
                @test maximum(abs, interior(dT)) > 0
                @test !any(isnan, interior(dT))
            end
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
