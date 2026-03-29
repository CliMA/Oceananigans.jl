include("reactant_test_utils.jl")
using Reactant: @trace
using Enzyme
using Statistics: mean

#####
##### Reactant single column model tests (Flat, Flat, Bounded)
#####

@testset "Reactant single column HydrostaticFreeSurfaceModel" begin
    arch = ReactantState()

    function run_timesteps!(model, Δt, Nt)
        @trace track_numbers=false for _ in 1:Nt
            time_step!(model, Δt)
        end
        return nothing
    end

    @testset "Construction" begin
        grid = RectilinearGrid(arch; size=16, z=(-200, 0), topology=(Flat, Flat, Bounded))
        model = HydrostaticFreeSurfaceModel(grid;
                    buoyancy = nothing,
                    tracers = :T)

        @test model isa HydrostaticFreeSurfaceModel
        @test model.grid.architecture isa ReactantState
    end

    @testset "Compiled time_step!" begin
        grid = RectilinearGrid(arch; size=16, z=(-200, 0), topology=(Flat, Flat, Bounded))
        model = HydrostaticFreeSurfaceModel(grid;
                    buoyancy = nothing,
                    tracers = :T)

        Δt = 60.0
        Nt = 4
        compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
        compiled_run!(model, Δt, Nt)
        @test model.clock.iteration == Nt
    end

    @testset "With BuoyancyTracer" begin
        grid = RectilinearGrid(arch; size=16, z=(-200, 0), topology=(Flat, Flat, Bounded))
        model = HydrostaticFreeSurfaceModel(grid;
                    buoyancy = BuoyancyTracer(),
                    tracers = :b)

        Δt = 60.0
        Nt = 4
        compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
        compiled_run!(model, Δt, Nt)
        @test model.clock.iteration == Nt
    end

    @testset "With TKEDissipationVerticalDiffusivity" begin
        grid = RectilinearGrid(arch; size=16, z=(-200, 0), topology=(Flat, Flat, Bounded))
        closure = TKEDissipationVerticalDiffusivity()
        model = HydrostaticFreeSurfaceModel(grid;
                    closure,
                    buoyancy = BuoyancyTracer(),
                    tracers = :b)

        b_init = CenterField(grid)
        set!(b_init, z -> 1e-5 * z)
        set!(model, b=b_init)

        Δt = 60.0
        Nt = 4
        compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
        compiled_run!(model, Δt, Nt)
        @test model.clock.iteration == Nt
    end

    @testset "Enzyme reverse-mode gradient" begin
        grid = RectilinearGrid(arch; size=16, z=(-200, 0), topology=(Flat, Flat, Bounded))
        model = HydrostaticFreeSurfaceModel(grid;
                    buoyancy = nothing,
                    tracers = :T)

        dmodel = Enzyme.make_zero(model)

        T_init  = CenterField(grid)
        set!(T_init, z -> 0.01 * z)
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

        Δt     = 60.0
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
