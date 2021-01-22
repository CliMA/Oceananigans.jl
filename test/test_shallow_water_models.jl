using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded

function time_stepping_shallow_water_model_works(arch, topo, coriolis)
    grid = RegularCartesianGrid(size=(1, 1, 1), extent=(2π, 2π, 2π), topology=topo)
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch, coriolis=coriolis)
    set!(model, h=1)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    run!(simulation)

    return model.clock.iteration == 1
end

function time_step_wizard_shallow_water_model_works(arch, topo, coriolis)
    grid = RegularCartesianGrid(size=(1, 1, 1), extent=(2π, 2π, 2π), topology=topo)
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch, coriolis=coriolis)
    set!(model, h=1)

    wizard = TimeStepWizard(cfl=1.0, Δt=1.0, max_change=1.1, max_Δt=10)

    simulation = Simulation(model, Δt=wizard, stop_iteration=1)
    run!(simulation)

    return model.clock.iteration == 1
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
        for arch in archs, FT in float_types
            N = (4, 4, 1)
            L = (2π, 3π, 5π)

            grid = RegularCartesianGrid(FT, size=N, extent=L)
            model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch)

            x, y, z = nodes((Face, Cell, Cell), model.grid, reshape=true)

            uh₀(x, y, z) = x * y^2
            uh_answer = @. x * y^2

            h₀ = rand(size(grid)...)
            h_answer = deepcopy(h₀)

            set!(model, uh=uh₀, h=h₀)

            uh, vh, h = model.solution

            @test all(interior(uh) .≈ uh_answer)
            @test all(interior(h) .≈ h_answer)
        end
    end

    for arch in archs
        for topo in topos
            @testset "Time-stepping ShallowWaterModels [$arch, $topo]" begin
                @info "  Testing time-stepping ShallowWaterModels [$arch, $topo]..."
                @test time_stepping_shallow_water_model_works(arch, topo, nothing)
            end
        end

        for coriolis in (nothing, FPlane(f=1), BetaPlane(f₀=1, β=0.1))
            @testset "Time-stepping ShallowWaterModels [$arch, $(typeof(coriolis))]" begin
                @info "  Testing time-stepping ShallowWaterModels [$arch, $(typeof(coriolis))]..."
                @test time_stepping_shallow_water_model_works(arch, topos[1], coriolis)
            end
        end

	@testset "Time-step Wizard ShallowWaterModels [$arch, $topos[1]]" begin
	    @info "  Testing time-step wizard ShallowWaterModels [$arch, $topos[1]]..."
            @test time_step_wizard_shallow_water_model_works(archs[1], topos[1], nothing)
	end	
    end
end
