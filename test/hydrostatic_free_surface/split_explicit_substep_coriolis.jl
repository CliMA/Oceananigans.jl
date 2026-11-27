include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: CoriolisForwardBackwardScheme,
                                                                                  ForwardBackwardScheme
using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.Coriolis: ActiveWeightedEnstrophyConserving, ActiveWeightedEnergyConserving, TriadScheme

function substep_coriolis_test_model(arch, timestepper, coriolis, scheme; H = 700, TX = Bounded, Δt = 3600)
    Lx = 50 * 0.3 * 111.13e3 * cosd(75)
    Ly = 100 * 0.1 * 111.13e3
    grid = RectilinearGrid(arch; size = (50, 100, 5), x = (0, Lx), y = (0, Ly), z = (-H, 0),
                           topology = (TX, Bounded, Bounded), halo = (6, 6, 6))

    free_surface = SplitExplicitFreeSurface(grid; cfl = 0.7, fixed_Δt = Δt, timestepper = scheme)

    model = HydrostaticFreeSurfaceModel(grid; free_surface, coriolis, timestepper,
                                        closure = (VerticalScalarDiffusivity(ν = 1e-3),
                                                   HorizontalScalarDiffusivity(ν = 600.0)),
                                        momentum_advection = nothing, tracer_advection = nothing,
                                        tracers = (), buoyancy = nothing)

    rng = Xoshiro(20260927)
    u₀ = 1e-6 * randn(rng, size(model.velocities.u))
    v₀ = 1e-6 * randn(rng, size(model.velocities.v))
    set!(model, u = u₀, v = v₀)

    return model
end

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: barotropic_x_coriolis, barotropic_y_coriolis,
                                                                                  horizontal_peripheral_node
using Oceananigans.Grids: static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ
using Oceananigans.Operators: Azᶠᶜᶜ, Azᶜᶠᶜ

@kernel function _barotropic_coriolis_work!(Wᵘ, Wᵛ, grid, coriolis, U, V, Uʳ, Vʳ)
    i, j = @index(Global, NTuple)
    k = grid.Nz + 1
    Hᶠᶜ = static_column_depthᶠᶜᵃ(i, j, grid)
    Hᶜᶠ = static_column_depthᶜᶠᵃ(i, j, grid)
    Cᵁ = barotropic_x_coriolis(i, j, k, grid, coriolis, V, Vʳ, Hᶠᶜ)
    Cⱽ = barotropic_y_coriolis(i, j, k, grid, coriolis, U, Uʳ, Hᶜᶠ)
    @inbounds Wᵘ[i, j, 1] = ifelse(Hᶠᶜ > 0, Azᶠᶜᶜ(i, j, k, grid) * U[i, j, 1] * Cᵁ / ifelse(Hᶠᶜ > 0, Hᶠᶜ, one(Hᶠᶜ)), zero(grid))
    @inbounds Wᵛ[i, j, 1] = ifelse(Hᶜᶠ > 0, Azᶜᶠᶜ(i, j, k, grid) * V[i, j, 1] * Cⱽ / ifelse(Hᶜᶠ > 0, Hᶜᶠ, one(Hᶜᶠ)), zero(grid))
end

# Zero the transports where the substep Coriolis is inactive, including the east and north walls
@kernel function _zero_inactive_transports!(U, V, grid)
    i, j = @index(Global, NTuple)
    @inbounds U[i, j, 1] *= !horizontal_peripheral_node(i, j, grid, Face(), Center()) & (static_column_depthᶠᶜᵃ(i, j, grid) > 0)
    @inbounds V[i, j, 1] *= !horizontal_peripheral_node(i, j, grid, Center(), Face()) & (static_column_depthᶜᶠᵃ(i, j, grid) > 0)
end

# Work of the barotropic Coriolis term on Σ Az (U² / Hᶠᶜ + V² / Hᶜᶠ) / 2, relative to the sum of its magnitudes
function relative_barotropic_coriolis_work(grid, coriolis; seed = 20261126)
    rng = Xoshiro(seed)
    U, Uʳ, Wᵘ = (Field{Face, Center, Nothing}(grid) for _ in 1:3)
    V, Vʳ, Wᵛ = (Field{Center, Face, Nothing}(grid) for _ in 1:3)
    set!(U, randn(rng, size(U)))
    set!(V, randn(rng, size(V)))
    launch!(architecture(grid), grid, (grid.Nx + 1, grid.Ny + 1), _zero_inactive_transports!, U, V, grid)
    fill_halo_regions!((U, V))
    launch!(architecture(grid), grid, :xy, _barotropic_coriolis_work!, Wᵘ, Wᵛ, grid, coriolis, U, V, Uʳ, Vʳ)
    wᵘ = Array(interior(Wᵘ))
    wᵛ = Array(interior(Wᵛ))
    return abs(sum(wᵘ) + sum(wᵛ)) / (sum(abs, wᵘ) + sum(abs, wᵛ))
end

function random_bathymetry_grid(arch, TX, immersed_boundary_type; seed = 20261126)
    rng = Xoshiro(seed)
    Nx, Ny = 16, 12
    bottom = - 8 .- 3992 .* rand(rng, Nx, Ny)
    bottom[rand(rng, Nx, Ny) .< 0.15] .= 0
    underlying_grid = RectilinearGrid(arch; size = (Nx, Ny, 4), x = (0, 4e5), y = (0, 3e5), z = (-4000, 0),
                                      topology = (TX, Bounded, Bounded))
    return ImmersedBoundaryGrid(underlying_grid, immersed_boundary_type(on_architecture(arch, bottom)))
end

velocity_norm(model) =sqrt(sum(abs2, Array(interior(model.velocities.u))) +
                            sum(abs2, Array(interior(model.velocities.v))))

function velocity_growth(model; Δt = 3600, Nwarmup = 30, Nsteps = 100)
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

        @testset "Uniform inertial oscillation on a periodic grid [$(typeof(arch))]" begin
            @info "  Testing substep Coriolis across periodic halos [$(typeof(arch))]..."
            for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
                grid = RectilinearGrid(arch; size = (4, 4, 4), extent = (1e5, 1e5, 450), topology = (Periodic, Periodic, Bounded))
                free_surface = SplitExplicitFreeSurface(grid; substeps = 30, timestepper = CoriolisForwardBackwardScheme())
                model = HydrostaticFreeSurfaceModel(grid; free_surface, timestepper, coriolis = FPlane(f = -1.409e-4),
                                                    momentum_advection = nothing, tracers = (), buoyancy = nothing)
                set!(model, u = 0.1)
                λ = velocity_growth(model; Nwarmup = 10, Nsteps = 50)
                @info "    $timestepper: λ = $λ"
                @test λ < 1
            end
        end

        @testset "Substep Coriolis follows the Coriolis scheme [$(typeof(arch))]" begin
            @info "  Testing CoriolisForwardBackwardScheme with every Coriolis scheme [$(typeof(arch))]..."
            for scheme in (EnstrophyConserving(), EnergyConserving(), ActiveWeightedEnstrophyConserving(),
                           ActiveWeightedEnergyConserving(), TriadScheme())
                coriolis = FPlane(f = -1.409e-4; scheme)
                λ_lagged = velocity_growth(substep_coriolis_test_model(arch, :QuasiAdamsBashforth2, coriolis, ForwardBackwardScheme()))
                λ_substep = velocity_growth(substep_coriolis_test_model(arch, :QuasiAdamsBashforth2, coriolis, CoriolisForwardBackwardScheme()))
                @info "    $(summary(scheme)): λ = $λ_lagged (lagged), $λ_substep (substep)"
                @test λ_substep < 1
                @test λ_substep < λ_lagged + 1e-4
            end
        end

        @testset "Energy-neutral barotropic Coriolis over varying depth [$(typeof(arch))]" begin
            @info "  Testing that EnergyConserving substep Coriolis does no work over random bathymetry [$(typeof(arch))]..."
            coriolis = BetaPlane(f₀ = 1e-4, β = 1e-11; scheme = EnergyConserving())
            for TX in (Bounded, Periodic), ib in (GridFittedBottom, PartialCellBottom)
                grid = random_bathymetry_grid(arch, TX, ib)
                r = relative_barotropic_coriolis_work(grid, coriolis)
                @info "    $TX, $ib: relative work = $r"
                @test r < 1e-12
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
