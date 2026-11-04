include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans: TendencyCallsite, TracerTendencyCallsite

add_to_tendency!(model, name) = interior(model.timestepper.Gⁿ[name]) .+= 1

# Add 1 to the c and u tendencies at their callsites for one time step, after which c = u = Δt
function test_tendency_callsites(model; Δt = 0.1)
    simulation = Simulation(model; Δt, stop_iteration = 1, verbose = false)
    add_callback!(simulation, model -> add_to_tendency!(model, :c), callsite = TracerTendencyCallsite())
    add_callback!(simulation, model -> add_to_tendency!(model, :u), callsite = TendencyCallsite())
    run!(simulation)

    for field in (model.tracers.c, model.velocities.u)
        @test on_architecture(CPU(), interior(field)) ≈ fill(Δt, size(field))
    end
end

@testset "Tendency callsites" begin
    for arch in archs
        grid = RectilinearGrid(arch; size = (2, 2, 2), extent = (1, 1, 1), topology = (Periodic, Periodic, Bounded))

        @testset "HydrostaticFreeSurfaceModel [$(typeof(arch)), $timestepper, $(nameof(typeof(free_surface)))]" for
                timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3),
                free_surface in (ExplicitFreeSurface(), ImplicitFreeSurface())

            test_tendency_callsites(HydrostaticFreeSurfaceModel(grid; timestepper, free_surface, tracers = :c, buoyancy = nothing))
        end

        # Both callsites follow all the tendencies
        @testset "NonhydrostaticModel [$(typeof(arch))]" begin
            test_tendency_callsites(NonhydrostaticModel(grid; tracers = :c))
        end
    end
end
