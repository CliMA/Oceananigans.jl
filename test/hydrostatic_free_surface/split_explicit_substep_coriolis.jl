include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: CoriolisForwardBackwardScheme,
                                                                                  ForwardBackwardScheme

function substep_coriolis_test_model(arch, timestepper, coriolis, scheme; H = 700, TX = Bounded, Δt = 3600)
    Lx = 50 * 0.3 * 111.13e3 * cosd(75)
    Ly = 100 * 0.1 * 111.13e3
    grid = RectilinearGrid(arch; size = (50, 100, 5), x = (0, Lx), y = (0, Ly), z = (-H, 0),
                           topology = (TX, Bounded, Bounded), halo = (6, 6, 6))

    free_surface = SplitExplicitFreeSurface(grid; cfl = 0.1, fixed_Δt = Δt, timestepper = scheme)

    model = HydrostaticFreeSurfaceModel(grid; free_surface, coriolis, timestepper,
                                        closure = (VerticalScalarDiffusivity(ν = 1e-3),
                                                   HorizontalScalarDiffusivity(ν = 600.0)),
                                        momentum_advection = nothing, tracer_advection = nothing,
                                        tracers = (), buoyancy = nothing)

    Random.seed!(20260927)
    set!(model, u = (x, y, z) -> 1e-6 * randn(), v = (x, y, z) -> 1e-6 * randn())

    return model
end

velocity_norm(model) = sqrt(sum(abs2, Array(interior(model.velocities.u))) +
                            sum(abs2, Array(interior(model.velocities.v))))

function velocity_growth(model; Δt = 3600, Nwarmup = 60, Nsteps = 200)
    for _ in 1:Nwarmup
        time_step!(model, Δt)
    end
    n₀ = velocity_norm(model)
    for _ in Nwarmup+1:Nsteps
        time_step!(model, Δt)
    end
    return (velocity_norm(model) / n₀)^(1 / (Nsteps - Nwarmup))
end

@testset "Split-explicit substep Coriolis" begin
    for arch in archs
        @testset "Inert without rotation [$(typeof(arch))]" begin
            @info "  Testing that CoriolisForwardBackwardScheme matches ForwardBackwardScheme without rotation [$(typeof(arch))]..."
            for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
                reference = substep_coriolis_test_model(arch, timestepper, nothing, ForwardBackwardScheme())
                corrected = substep_coriolis_test_model(arch, timestepper, nothing, CoriolisForwardBackwardScheme())
                for _ in 1:10
                    time_step!(reference, 3600)
                    time_step!(corrected, 3600)
                end
                @test Array(interior(corrected.velocities.u)) == Array(interior(reference.velocities.u))
                @test Array(interior(corrected.free_surface.displacement)) == Array(interior(reference.free_surface.displacement))
            end
        end

        @testset "Lagged barotropic Coriolis mode [$(typeof(arch))]" begin
            @info "  Testing that substep Coriolis removes the lagged barotropic Coriolis mode [$(typeof(arch))]..."
            coriolis = FPlane(f = -1.409e-4)
            for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3), TX in (Bounded, Periodic)
                λ_lagged = velocity_growth(substep_coriolis_test_model(arch, timestepper, coriolis, ForwardBackwardScheme(); TX))
                λ_substep = velocity_growth(substep_coriolis_test_model(arch, timestepper, coriolis, CoriolisForwardBackwardScheme(); TX))
                @info "    $timestepper, $TX: λ = $λ_lagged (lagged), $λ_substep (substep)"
                @test λ_substep < 1
                @test λ_substep < λ_lagged + 1e-4
            end
        end

        @testset "Checkpointing [$(typeof(arch))]" begin
            model = substep_coriolis_test_model(arch, :QuasiAdamsBashforth2, FPlane(f = 1e-4), CoriolisForwardBackwardScheme())
            time_step!(model, 3600)
            state = Oceananigans.prognostic_state(model.free_surface.timestepper)
            restored = substep_coriolis_test_model(arch, :QuasiAdamsBashforth2, FPlane(f = 1e-4), CoriolisForwardBackwardScheme())
            Oceananigans.restore_prognostic_state!(restored.free_surface.timestepper, state)
            @test Array(interior(restored.free_surface.timestepper.previous_transports.U)) ==
                  Array(interior(model.free_surface.timestepper.previous_transports.U))
        end
    end
end
