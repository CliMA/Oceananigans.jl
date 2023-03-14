include("dependencies_for_runtests.jl")

using Oceananigans.Models.ShallowWaterModels
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

function time_stepping_shallow_water_model_works(arch, topo, coriolis, advection; timestepper=:RungeKutta3)
    grid = RectilinearGrid(arch, size=(1, 1), extent=(2π, 2π), topology=topo)
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, coriolis=coriolis,
                              momentum_advection=advection, timestepper=:RungeKutta3)
    set!(model, h=1)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    run!(simulation)

    return model.clock.iteration == 1
end

function time_step_wizard_shallow_water_model_works(arch, topo, coriolis)
    grid = RectilinearGrid(arch, size=(1, 1), extent=(2π, 2π), topology=topo)
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, coriolis=coriolis)
    set!(model, h=1)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=10)
    simulation.callbacks[:wizard] = Callback(wizard)
    run!(simulation)

    return model.clock.iteration == 1
end

function shallow_water_model_tracers_and_forcings_work(arch)
    grid = RectilinearGrid(arch, size=(1, 1), extent=(2π, 2π), topology=((Periodic, Periodic, Flat)))
    model = ShallowWaterModel(grid=grid, gravitational_acceleration=1, tracers=(:c, :d))
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

function test_shallow_water_diffusion_cosine(grid, formulation, fieldname, ξ) 
    κ, m = 1, 2 # diffusivity and cosine wavenumber

    closure = ShallowWaterScalarDiffusivity(ν = κ)
    momentum_advection = nothing
    tracer_advection = nothing
    mass_advection = nothing
    model = ShallowWaterModel(; grid, closure, 
                                gravitational_acceleration=1.0, 
                                momentum_advection, tracer_advection, mass_advection,
                                formulation)

    field = model.velocities[fieldname]
    interior(field) .= arch_array(architecture(grid), cos.(m * ξ))
    update_state!(model)

    # Step forward with small time-step relative to diff. time-scale
    Δt = 1e-6 * grid.Lx^2 / κ
    for n = 1:5
        time_step!(model, Δt)
    end

    diffusing_cosine(ξ, t, κ, m) = exp(-κ * m^2 * t) * cos(m * ξ)
    analytical_solution = Field(location(field), grid)
    analytical_solution .= diffusing_cosine.(ξ, model.clock.time, κ, m)

    return isapprox(field, analytical_solution, atol=1e-6, rtol=1e-6)
end

@testset "Shallow Water Models" begin
    @info "Testing shallow water models..."

    @testset "Must be Flat in the vertical" begin
        grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
        @test_throws ArgumentError ShallowWaterModel(grid=grid, gravitational_acceleration=1)

        grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
        @test_throws ArgumentError ShallowWaterModel(grid=grid, gravitational_acceleration=1)
    end

    @testset "Model constructor errors" begin
        grid = RectilinearGrid(size=(1, 1), extent=(1, 1), topology=(Periodic,Periodic,Flat))
        @test_throws MethodError ShallowWaterModel(architecture=CPU, grid=grid, gravitational_acceleration=1)
        @test_throws MethodError ShallowWaterModel(architecture=GPU, grid=grid, gravitational_acceleration=1)
    end

    topo = ( Flat,      Flat,     Flat )
   
    @testset "$topo model construction" begin
    @info "  Testing $topo model construction..."
        for arch in archs, FT in float_types
            grid = RectilinearGrid(arch, FT, topology=topo, size=(), extent=())
            model = ShallowWaterModel(grid=grid, gravitational_acceleration=1) 

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
        
                grid = RectilinearGrid(arch, FT, topology=topo, size=1, extent=1, halo=3)
                model = ShallowWaterModel(grid=grid, gravitational_acceleration=1) 
                
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
               #arch isa GPU && topo == (Bounded, Bounded, Flat) && continue

                grid = RectilinearGrid(arch, FT, topology=topo, size=(1, 1), extent=(1, 2), halo=(3, 3))
                model = ShallowWaterModel(grid=grid, gravitational_acceleration=1)

                @test model isa ShallowWaterModel
            end
        end
    end

    @testset "Setting ShallowWaterModel fields" begin
    @info "  Testing setting shallow water model fields..."
        for arch in archs, FT in float_types
            N = (4,   4)
            L = (2π, 3π)

            grid = RectilinearGrid(arch, FT, size=N, extent=L, topology=(Periodic, Periodic, Flat), halo=(3, 3))
            model = ShallowWaterModel(grid=grid, gravitational_acceleration=1)

            x, y, z = nodes(model.grid, (Face(), Center(), Center()), reshape=true)

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
        for advection in (nothing, CenteredSecondOrder(), WENO())
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

        @testset "ShallowWaterModel viscous diffusion [$arch]" begin
            Nx, Ny = 10, 12
            grid_x = RectilinearGrid(arch, size = Nx, x = (0, 1), topology = (Bounded, Flat, Flat))
            grid_y = RectilinearGrid(arch, size = Ny, y = (0, 1), topology = (Flat, Bounded, Flat))
            coords = (reshape(xnodes(grid_x, Face()), (Nx+1, 1)), reshape(ynodes(grid_y, Face()), (1, Ny+1)))
            
            for (fieldname, grid, coord) in zip([:u, :v], [grid_x, grid_y], coords)
                for formulation in (ConservativeFormulation(), VectorInvariantFormulation())
                    @info "  Testing ShallowWaterModel cosine viscous diffusion [$fieldname, $formulation]"
                    test_shallow_water_diffusion_cosine(grid, formulation, fieldname, coord)
                end
            end
        end
    end

    @testset "ShallowWaterModels with ImmersedBoundaryGrid" begin
        for arch in archs
            @testset "ShallowWaterModels with ImmersedBoundaryGrid [$arch]" begin
                @info "Testing ShallowWaterModels with ImmersedBoundaryGrid [$arch]"

                grid = RectilinearGrid(arch, size=(8, 8), x=(-10, 10), y=(0, 5), topology=(Periodic, Bounded, Flat))
                
                # Gaussian bump of width "1"
                bump(x, y, z) = y < exp(-x^2)
                
                grid_with_bump = ImmersedBoundaryGrid(grid, GridFittedBoundary(bump))
                model = ShallowWaterModel(grid=grid_with_bump, gravitational_acceleration=1)
                
                set!(model, h=1)
                simulation = Simulation(model, Δt=1.0, stop_iteration=1)
                run!(simulation)

                @test model.clock.iteration == 1
            end
        end
    end
end
