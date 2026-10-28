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
        (topology = (Periodic, Periodic, Periodic), size = (4, 4, 4), extent = (1, 1, 1),
         timestepper = :RungeKutta3, name = "3D (Periodic, Periodic, Periodic) — RK3"),
        (topology = (Periodic, Periodic, Flat), size = (4, 4), extent = (1, 1),
         timestepper = :QuasiAdamsBashforth2, name = "2D (Periodic, Periodic, Flat) — QB2"),
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
                set!(T_init, (x, y, z...) -> 0.01 * x + 0.01 * y)
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
end
