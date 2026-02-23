include("reactant_test_utils.jl")
using Reactant: @trace
using Enzyme
using CUDA
using Statistics: mean

#####
##### Reactant HydrostaticFreeSurfaceModel tests (ExplicitFreeSurface, FFT-free)
#####
# Blueprint phases (a)+(c)+(d): construction, compiled time-stepping with raise=true,
# and Enzyme reverse-mode differentiation.
# HFSM requires Bounded in z (free surface); we only test z=Bounded topologies.
# See test-blueprint.md in cursor-toolchain/rules/domains/differentiability/.

@testset "Reactant HydrostaticFreeSurfaceModel (ExplicitFreeSurface)" begin
    @info "Testing Reactant HydrostaticFreeSurfaceModel (ExplicitFreeSurface)..."
    arch = ReactantState()

    function run_timesteps!(model, Δt, Nt)
        @trace track_numbers=false for _ in 1:Nt
            time_step!(model, Δt)
        end
        return nothing
    end

    topologies = [
        ((Periodic, Periodic, Bounded), (4, 4, 4), (1, 1, 1)),
        ((Bounded,  Bounded,  Bounded), (4, 4, 4), (1, 1, 1)),
    ]

    for (topo, sz, ext) in topologies
        topo_str = join(nameof.(typeof.(topo)), ", ")

        @testset "($topo_str)" begin
            @info "  Testing ($topo_str)..."
            grid = RectilinearGrid(arch; size=sz, extent=ext, topology=topo)
            model = HydrostaticFreeSurfaceModel(grid;
                        free_surface = ExplicitFreeSurface(),
                        timestepper  = :QuasiAdamsBashforth2,
                        buoyancy     = nothing,
                        tracers      = :T)

            @testset "Construction" begin
                @test model isa HydrostaticFreeSurfaceModel
                @test model.grid.architecture isa ReactantState
                @test model.free_surface isa ExplicitFreeSurface
            end

            @testset "Compiled time_step! (raise=true)" begin
                @info "    Compiling and running time_step!..."
                Δt = 0.001
                Nt = 4
                compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
                compiled_run!(model, Δt, Nt)
                @test model.clock.iteration == Nt
            end

            @testset "Enzyme reverse-mode gradient" begin
                @info "    Compiling and running Enzyme gradient..."
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
end
