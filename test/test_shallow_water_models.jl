using Oceananigans
using Oceananigans.Models
using Oceananigans.Grids

function time_stepping_shallow_water_model_works(arch, topo, coriolis, advection; timestepper=:RungeKutta3)
    grid = RegularRectilinearGrid(size=(1, 1), extent=(2π, 2π), topology=topo)
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch, coriolis=coriolis,
                              advection=advection, timestepper=:RungeKutta3)
    set!(model, h=1)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    run!(simulation)

    return model.clock.iteration == 1
end

function time_step_wizard_shallow_water_model_works(arch, topo, coriolis)
    grid = RegularRectilinearGrid(size=(1, 1), extent=(2π, 2π), topology=topo)
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch, coriolis=coriolis)
    set!(model, h=1)

    wizard = TimeStepWizard(cfl=1.0, Δt=1.0, max_change=1.1, max_Δt=10)

    simulation = Simulation(model, Δt=wizard, stop_iteration=1)
    run!(simulation)

    return model.clock.iteration == 1
end

function shallow_water_model_tracers_and_forcings_work(arch)
    grid = RegularRectilinearGrid(size=(1, 1), extent=(2π, 2π), topology=((Periodic, Periodic, Flat)))
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

@testset "Shallow Water Models" begin
    @info "Testing shallow water models..."

    @testset "Must be Flat in the vertical" begin
        grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1), topology=(Periodic,Periodic,Bounded))
        @test_throws TypeError ShallowWaterModel(architecture=CPU, grid=grid, gravitational_acceleration=1)        

        grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1), topology=(Periodic,Periodic,Periodic))
        @test_throws TypeError ShallowWaterModel(architecture=CPU, grid=grid, gravitational_acceleration=1)        
    end

    @testset "Model constructor errors" begin
        grid = RegularRectilinearGrid(size=(1, 1), extent=(1, 1), topology=((Periodic,Periodic,Flat)))
        @test_throws TypeError ShallowWaterModel(architecture=CPU, grid=grid, gravitational_acceleration=1)
        @test_throws TypeError ShallowWaterModel(architecture=GPU, grid=grid, gravitational_acceleration=1)
    end

    topo = ( Flat,      Flat,     Flat )
   
    @testset "$topo model construction" begin
    @info "  Testing $topo model construction..."
        for arch in archs, FT in float_types                
            grid = RegularRectilinearGrid(FT, topology=topo, size=(), extent=())
            model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch) 
            
            @test model isa ShallowWaterModel
        end
    end

    topos = (
             (Bounded,   Flat,     Flat),    
             (Flat,      Bounded,  Flat),
            )

    for topo in topos
        @testset "$topo model construction" begin
            @info "  Testing $topo model construction..."
            for arch in archs, FT in float_types
                #arch isa GPU && topo == (Flat, Bounded, Flat) && continue
        
                grid = RegularRectilinearGrid(FT, topology=topo, size=(1), extent=(1))
                model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch) 
                
                @test model isa ShallowWaterModel
            end
        end
    end

    topos = (
             (Periodic, Periodic,  Flat),
             (Periodic,  Bounded,  Flat),
             (Bounded,   Bounded,  Flat),
            )

    for topo in topos
        @testset "$topo model construction" begin
            @info "  Testing $topo model construction..."
            for arch in archs, FT in float_types
		        arch isa GPU && topo == (Bounded, Bounded, Flat) && continue

                grid = RegularRectilinearGrid(FT, topology=topo, size=(1, 1), extent=(1, 2))
                model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch)

                @test model isa ShallowWaterModel
            end
        end
    end

    @testset "Setting ShallowWaterModel fields" begin
    @info "  Testing setting shallow water model fields..."
        for arch in archs, FT in float_types
            N = (4,   4)
            L = (2π, 3π)

            grid = RegularRectilinearGrid(FT, size=N, extent=L, topology=((Periodic, Periodic, Flat)))
            model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, architecture=arch)

            x, y, z = nodes((Face, Center, Center), model.grid, reshape=true)

            uh₀(x, y, z) = x * y^2
            uh_answer = @. x * y^2

            h₀ = rand(size(grid)...)
            h_answer = deepcopy(h₀)

            set!(model, uh=uh₀, h=h₀)

            uh, vh, h = model.solution

            @test all(Array(interior(uh)) .≈ uh_answer)
            @test all(Array(interior(h)) .≈ h_answer)
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
            @testset "Time-stepping ShallowWaterModels [$arch, $(typeof(coriolis))]" begin
                @info "  Testing time-stepping ShallowWaterModels [$arch, $(typeof(coriolis))]..."
                @test time_stepping_shallow_water_model_works(arch, topos[1], coriolis, nothing)
            end
        end

        @testset "Time-step Wizard ShallowWaterModels [$arch, $topos[1]]" begin
	    @info "  Testing time-step wizard ShallowWaterModels [$arch, $topos[1]]..."
            @test time_step_wizard_shallow_water_model_works(archs[1], topos[1], nothing)
        end

        # Advection = nothing is broken as halo does not have a maximum
        for advection in (nothing, CenteredSecondOrder(), WENO5())
            @testset "Time-stepping ShallowWaterModels [$arch, $(typeof(advection))]" begin
                @info "  Testing time-stepping ShallowWaterModels [$arch, $(typeof(advection))]..."
                @test time_stepping_shallow_water_model_works(arch, topos[1], nothing, advection)
            end
        end

        for timestepper in (:RungeKutta3, :QuasiAdamsBashforth2)
            @testset "Time-stepping ShallowWaterModels [$arch, $timestepper]" begin
                @info "  Testing time-stepping ShallowWaterModels [$arch, $timestepper]..."
                @test time_stepping_shallow_water_model_works(arch, topos[1], nothing, nothing, timestepper=timestepper)
            end
        end

        @testset "ShallowWaterModel with tracers and forcings [$arch]" begin
            @info "  Testing ShallowWaterModel with tracers and forcings [$arch]..."
            shallow_water_model_tracers_and_forcings_work(arch)
        end
    end
end
