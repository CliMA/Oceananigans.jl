include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans: TendencyCallsite, TracerTendencyCallsite, MomentumTendency, TracerTendency

add_to_tendency!(model, name) = interior(model.timestepper.Gⁿ[name]) .+= 1

function tendency_callsite_simulation(model; Δt)
    simulation = Simulation(model; Δt, stop_iteration = 1, verbose = false)

    add_callback!(simulation, model -> add_to_tendency!(model, :c), callsite = TracerTendencyCallsite())
    add_callback!(simulation, model -> add_to_tendency!(model, :u), callsite = TendencyCallsite())

    return simulation
end

@testset "Tendency callsites" begin
    @test TendencyCallsite() isa TendencyCallsite{MomentumTendency}
    @test TracerTendencyCallsite() === TendencyCallsite(TracerTendency())

    Δt = 0.1

    for arch in archs
        grid = RectilinearGrid(arch; size = (2, 2, 2), extent = (1, 1, 1), topology = (Periodic, Periodic, Bounded))

        @testset "HydrostaticFreeSurfaceModel [$(typeof(arch)), $timestepper, $(nameof(typeof(free_surface)))]" for
                timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3),
                free_surface in (ExplicitFreeSurface(), ImplicitFreeSurface())

            model = HydrostaticFreeSurfaceModel(grid; timestepper, free_surface, tracers = :c, buoyancy = nothing)
            run!(tendency_callsite_simulation(model; Δt))

            @test on_architecture(CPU(), interior(model.tracers.c)) ≈ fill(Δt, size(model.tracers.c))
            @test on_architecture(CPU(), interior(model.velocities.u)) ≈ fill(Δt, size(model.velocities.u))
        end

        @testset "NonhydrostaticModel [$(typeof(arch))]" begin
            model = NonhydrostaticModel(grid; tracers = :c)
            run!(tendency_callsite_simulation(model; Δt))

            # Both callsites follow all the tendencies
            for field in (model.tracers.c, model.velocities.u)
                @test on_architecture(CPU(), interior(field)) ≈ fill(Δt, size(field))
            end
        end
    end
end
