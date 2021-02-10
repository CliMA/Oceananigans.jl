using Oceananigans: CPU, GPU
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Oceananigans.Grids: Periodic, Bounded

function time_stepping_hydrostatic_free_surface_model_works(arch, topo, coriolis)
    grid = RegularCartesianGrid(size=(1, 1, 1), extent=(2π, 2π, 2π), topology=topo)
    model = HydrostaticFreeSurfaceModel(grid=grid, architecture=arch, coriolis=coriolis)
    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    run!(simulation)

    return model.clock.iteration == 1
end

function hydrostatic_free_surface_model_tracers_and_forcings_work(arch)
    grid = RegularCartesianGrid(size=(1, 1, 1), extent=(2π, 2π, 2π))
    model = HydrostaticFreeSurfaceModel(grid=grid, architecture=arch, tracers=(:T, :S, :c, :d))
    set!(model, η=1)

    @test model.tracers.T isa Field
    @test model.tracers.S isa Field
    @test model.tracers.c isa Field
    @test model.tracers.d isa Field

    @test haskey(model.forcing, :u)
    @test haskey(model.forcing, :v)
    @test haskey(model.forcing, :η)
    @test haskey(model.forcing, :T)
    @test haskey(model.forcing, :S)
    @test haskey(model.forcing, :c)
    @test haskey(model.forcing, :d)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
    run!(simulation)

    @test model.clock.iteration == 1

    return nothing
end

@testset "Hydrostatic free surface Models" begin
    @info "Testing hydrostatic free surface models..."

    @testset "Model constructor errors" begin
        grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1))
        @test_throws TypeError HydrostaticFreeSurfaceModel(architecture=CPU, grid=grid)
        @test_throws TypeError HydrostaticFreeSurfaceModel(architecture=GPU, grid=grid)
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
                grid = RegularCartesianGrid(FT, topology=topo, size=(1, 1, 1), extent=(1, 2, 3))
                model = HydrostaticFreeSurfaceModel(grid=grid, architecture=arch)

                # Just testing that the model was constructed with no errors/crashes.
                @test model isa HydrostaticFreeSurfaceModel

                # Test that the grid didn't get mangled (sort of)
                @test size(grid) === size(model.grid)
            end
        end
    end

    @testset "Setting HydrostaticFreeSurfaceModel fields" begin
        @info "  Testing setting hydrostatic free surface model fields..."
        for arch in archs, FT in float_types
            N = (4, 4, 1)
            L = (2π, 3π, 5π)

            grid = RegularCartesianGrid(FT, size=N, extent=L)
            model = HydrostaticFreeSurfaceModel(grid=grid, architecture=arch)

            x, y, z = nodes((Face, Center, Center), model.grid, reshape=true)

            u₀(x, y, z) = x * y^2
            u_answer = @. x * y^2

            η₀ = rand(size(grid)...)
            η_answer = deepcopy(η₀)

            set!(model, u=u₀, η=η₀)

            u, v, w = model.velocities
            η = model.free_surface.η

            @test all(interior(u) .≈ u_answer)
            @test all(interior(η) .≈ η_answer)
        end
    end

    for arch in archs
        for topo in topos
            @testset "Time-stepping HydrostaticFreeSurfaceModels [$arch, $topo]" begin
                @info "  Testing time-stepping HydrostaticFreeSurfaceModels [$arch, $topo]..."
                @test time_stepping_hydrostatic_free_surface_model_works(arch, topo, nothing)
            end
        end

        for coriolis in (nothing, FPlane(f=1), BetaPlane(f₀=1, β=0.1))
            @testset "Time-stepping HydrostaticFreeSurfaceModels [$arch, $(typeof(coriolis))]" begin
                @info "  Testing time-stepping HydrostaticFreeSurfaceModels [$arch, $(typeof(coriolis))]..."
                @test time_stepping_hydrostatic_free_surface_model_works(arch, topos[1], coriolis)
            end
        end

        @testset "HydrostaticFreeSurfaceModel with tracers and forcings [$arch]" begin
            @info "  Testing HydrostaticFreeSurfaceModel with tracers and forcings [$arch]..."
            hydrostatic_free_surface_model_tracers_and_forcings_work(arch)
        end
    end
end
