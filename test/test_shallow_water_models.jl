using Oceananigans.Models.ShallowWaterModels: ConservativeSolution, PrimitiveSolutionLinearizedHeight
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded

function time_stepping_shallow_water_model_works(arch, topo, coriolis, advection;
                                                 timestepper=:RungeKutta3, solution=ConservativeSolution())

    grid = RegularCartesianGrid(size=(1, 1, 1), extent=(2π, 2π, 2π), topology=topo)

    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch, coriolis=coriolis,
                              solution=solution, advection=advection, timestepper=:RungeKutta3)

    if solution isa ConservativeSolution
        set!(model, h=1)
    elseif solution isa PrimitiveSolutionLinearizedHeight
        set!(model, η=1)
    end

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    run!(simulation)

    return model.clock.iteration == 1
end

function time_step_wizard_shallow_water_model_works(arch, topo, coriolis, solution=ConservativeSolution())
    grid = RegularCartesianGrid(size=(1, 1, 1), extent=(2π, 2π, 2π), topology=topo)
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch, solution=solution, coriolis=coriolis)

    if solution isa ConservativeSolution
        set!(model, h=1)
    elseif solution isa PrimitiveSolutionLinearizedHeight
        set!(model, η=1)
    end

    wizard = TimeStepWizard(cfl=1.0, Δt=1.0, max_change=1.1, max_Δt=10)

    simulation = Simulation(model, Δt=wizard, stop_iteration=1)
    run!(simulation)

    return model.clock.iteration == 1
end

function conservative_shallow_water_model_tracers_and_forcings_work(arch)
    grid = RegularCartesianGrid(size=(1, 1, 1), extent=(2π, 2π, 2π))
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch, tracers=(:c, :d))
    set!(model, h=1)

    @test model.tracers.c isa Field
    @test model.tracers.d isa Field

    @test haskey(model.forcing, :uh)
    @test haskey(model.forcing, :vh)
    @test haskey(model.forcing, :h)
    @test haskey(model.forcing, :c)
    @test haskey(model.forcing, :d)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    run!(simulation)

    @test model.clock.iteration == 1

    return nothing
end

function primitive_shallow_water_model_tracers_and_forcings_work(arch)
    grid = RegularCartesianGrid(size=(1, 1, 1), extent=(2π, 2π, 2π))
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch, tracers=(:c, :d),
                              solution=PrimitiveSolutionLinearizedHeight())

    @test model.tracers.c isa Field
    @test model.tracers.d isa Field

    @test haskey(model.forcing, :u)
    @test haskey(model.forcing, :v)
    @test haskey(model.forcing, :η)
    @test haskey(model.forcing, :c)
    @test haskey(model.forcing, :d)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    run!(simulation)

    @test model.clock.iteration == 1

    return nothing
end

@testset "Shallow Water Models" begin
    @info "Testing shallow water models..."

    @testset "Model constructor errors" begin
        grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1))
        @test_throws TypeError ShallowWaterModel(architecture=CPU, grid=grid, gravitational_acceleration=1)
        @test_throws TypeError ShallowWaterModel(architecture=GPU, grid=grid, gravitational_acceleration=1)
    end

    topos = (
             (Periodic, Periodic,  Bounded),
             (Periodic,  Bounded,  Bounded),
             (Bounded,   Bounded,  Bounded),
            )

    for topo in topos
        @testset "$topo model construction" begin
            @info "  Testing $topo model construction..."
            for arch in archs, FT in float_types
		        arch isa GPU && topo == (Bounded, Bounded, Bounded) && continue

                grid = RegularCartesianGrid(FT, topology=topo, size=(1, 1, 1), extent=(1, 2, 3))
                model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch)

                # Just testing that the model was constructed with no errors/crashes.
                @test model isa ShallowWaterModel

                # Test that the grid didn't get mangled (sort of)
                @test size(grid) === size(model.grid)

                too_big_grid = RegularCartesianGrid(FT, topology=topo, size=(1, 1, 2), extent=(1, 2, 3))

                @test_throws ArgumentError ShallowWaterModel(grid=too_big_grid, gravitational_acceleration=1, architecture=arch)
            end
        end
    end

    @testset "Setting ShallowWaterModel fields" begin
        @info "  Testing setting shallow water model fields..."
        # ConservativeSolution
        for arch in archs, FT in float_types
            N = (4, 4, 1)
            L = (2π, 3π, 5π)

            grid = RegularCartesianGrid(FT, size=N, extent=L)
            model = ShallowWaterModel(grid=grid, gravitational_acceleration=1,
                                      solution=ConservativeSolution(), architecture=arch)

            x, y, z = nodes((Face, Center, Center), model.grid, reshape=true)

            uh₀(x, y, z) = x * y^2
            uh_answer = @. x * y^2

            h₀ = rand(size(grid)...)
            h_answer = deepcopy(h₀)

            set!(model, uh=uh₀, h=h₀)

            uh, vh, h = model.solution

            @test all(interior(uh) .≈ uh_answer)
            @test all(interior(h) .≈ h_answer)
        end

        # PrimitiveSolutionLinearizedHeight
        for arch in archs, FT in float_types
            N = (4, 4, 1)
            L = (2π, 3π, 5π)

            grid = RegularCartesianGrid(FT, size=N, extent=L)
            model = ShallowWaterModel(grid=grid, gravitational_acceleration=1,
                                      solution=PrimitiveSolutionLinearizedHeight(), architecture=arch)

            x, y, z = nodes((Face, Center, Center), model.grid, reshape=true)

            u₀(x, y, z) = x * y^2
            u_answer = @. x * y^2

            η₀ = rand(size(grid)...)
            η_answer = deepcopy(η₀)

            set!(model, u=u₀, η=η₀)

            u, v, η = model.solution

            @test all(interior(u) .≈ u_answer)
            @test all(interior(η) .≈ η_answer)
        end

    end

    for arch in archs
        for topo in topos
            @testset "Time-stepping ShallowWaterModels [$arch, $topo]" begin
                @info "  Testing time-stepping ShallowWaterModels [$arch, $topo]..."
                @test time_stepping_shallow_water_model_works(arch, topo, nothing, nothing)
            end
        end

        for coriolis in (nothing, FPlane(f=1), BetaPlane(f₀=1, β=0.1))
            for solution in (ConservativeSolution(), PrimitiveSolutionLinearizedHeight())
                @testset "Time-stepping ShallowWaterModels [$arch, $(typeof(coriolis)), $(typeof(solution))]" begin
                    @info "  Testing time-stepping ShallowWaterModels [$arch, $(typeof(coriolis)), $(typeof(solution))]..."
                    @test time_stepping_shallow_water_model_works(arch, topos[1], coriolis, nothing)
                end
            end
        end

        for solution in (ConservativeSolution(), PrimitiveSolutionLinearizedHeight())
            @testset "Time-step Wizard ShallowWaterModels [$arch, $topos[1], $(typeof(solution))]" begin
                @info "  Testing time-step wizard ShallowWaterModels [$arch, $topos[1], $(typeof(solution))]..."
                @test time_step_wizard_shallow_water_model_works(archs[1], topos[1], nothing, solution)
            end
        end
                
        for advection in (nothing, CenteredSecondOrder(), WENO5())
            for solution in (ConservativeSolution(), PrimitiveSolutionLinearizedHeight())
                @testset "Time-stepping ShallowWaterModels [$arch, $(typeof(advection)), $(typeof(solution))]" begin
                    @info "  Testing time-stepping ShallowWaterModels [$arch, $(typeof(advection)), $(typeof(solution))]..."
                    @test time_stepping_shallow_water_model_works(arch, topos[1], nothing, advection, solution=solution)
                end
            end
        end

        for timestepper in (:RungeKutta3, :QuasiAdamsBashforth2)
            for solution in (ConservativeSolution(), PrimitiveSolutionLinearizedHeight())
                @testset "Time-stepping ShallowWaterModels [$arch, $timestepper, $(typeof(solution))]" begin
                    @info "  Testing time-stepping ShallowWaterModels [$arch, $timestepper, $(typeof(solution))]..."
                    @test time_stepping_shallow_water_model_works(arch, topos[1], nothing, nothing,
                                                                  timestepper=timestepper, solution=solution)
                end
            end
        end

        @testset "ShallowWaterModel with tracers and forcings [$arch]" begin
            @info "  Testing ShallowWaterModel with tracers and forcings [$arch]..."
            conservative_shallow_water_model_tracers_and_forcings_work(arch)
            primitive_shallow_water_model_tracers_and_forcings_work(arch)
        end
    end
end
