include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans: TendencyCallsite
using Oceananigans.BoundaryConditions: needs_implicit_solver

using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection,
                              update_advection!,
                              adaptive_advection_timestep,
                              advective_tracer_flux_z,
                              implicit_advection_upper_diagonal,
                              implicit_advection_lower_diagonal,
                              implicit_advection_diagonal
using Oceananigans.Grids: Center, Face, znode
using Oceananigans.Operators: volume, ℑzᵃᵃᶠ
using Oceananigans.TimeSteppers: AdaptiveVerticallyImplicitDiscretization, ExplicitTimeDiscretization,
                                 RungeKutta3TimeStepper, time_discretization, implicit_step!, reset!
using Oceananigans.TurbulenceClosures: implicit_diffusion_solver, VerticallyImplicitTimeDiscretization

@testset "AdaptiveVerticallyImplicitDiscretization construction" begin
    td = AdaptiveVerticallyImplicitDiscretization(cfl=0.3)
    @test td.cfl == 0.3
    @test td.Δt[] == zero(td.cfl)

    td_f32 = AdaptiveVerticallyImplicitDiscretization(Float32; cfl=0.4)
    @test td_f32.cfl isa Float32
    @test td_f32.cfl == 0.4f0
end

@testset "AIVA scheme dispatch" begin
    for scheme in (WENO, Centered, UpwindBiased)
        extra_kw = scheme == WENO ? (; weight_computation = Oceananigans.Utils.NormalDivision) : (; )
        explicit = scheme(; extra_kw...)
        adaptive = scheme(; time_discretization = AdaptiveVerticallyImplicitDiscretization(), extra_kw...)

        @test !(explicit isa AdaptiveImplicitVerticalAdvection)
        @test adaptive isa AdaptiveImplicitVerticalAdvection

        @test time_discretization(explicit) isa ExplicitTimeDiscretization
        @test time_discretization(adaptive) isa AdaptiveVerticallyImplicitDiscretization

        @test !needs_implicit_solver(explicit)
        @test  needs_implicit_solver(adaptive)
    end
end

@testset "AIVA flux equals explicit when α ≪ cfl" begin
    grid = RectilinearGrid(CPU(), size=(1, 1, 8),
                           x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Bounded))

    Δt = 1e-3
    aiva_td = AdaptiveVerticallyImplicitDiscretization(cfl=10.0)
    aiva_td.Δt[] = Δt

    explicit_scheme = WENO(; weight_computation = Oceananigans.Utils.NormalDivision)
    adaptive_scheme = WENO(; time_discretization = aiva_td, weight_computation = Oceananigans.Utils.NormalDivision)

    W = ZFaceField(grid)
    set!(W, (x, y, z) -> 1.0)
    fill_halo_regions!(W)

    c = CenterField(grid)
    set!(c, (x, y, z) -> sinpi(z))
    fill_halo_regions!(c)

    Nz = size(grid, 3)
    for k in 2:Nz
        flux_explicit = advective_tracer_flux_z(1, 1, k, grid, explicit_scheme, ExplicitTimeDiscretization(), W, c)
        flux_adaptive = advective_tracer_flux_z(1, 1, k, grid, adaptive_scheme, aiva_td, W, c)
        @test flux_adaptive ≈ flux_explicit rtol=1e-12
    end
end

@testset "AIVA flux is CFL-limited when α > cfl" begin
    grid = RectilinearGrid(CPU(), size=(1, 1, 8),
                           x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Bounded))

    cfl = 0.3
    Δt  = 1.0
    aiva_td = AdaptiveVerticallyImplicitDiscretization(cfl=cfl)
    aiva_td.Δt[] = Δt

    scheme = WENO(; time_discretization = aiva_td, weight_computation = Oceananigans.Utils.NormalDivision)

    W = ZFaceField(grid)
    set!(W, (x, y, z) -> 1.0)
    fill_halo_regions!(W)

    c = CenterField(grid)
    set!(c, (x, y, z) -> sinpi(z))
    fill_halo_regions!(c)

    Δz_face = 1 / size(grid, 3)
    α_expected = abs(1.0) * Δt / Δz_face
    expected_scale = cfl / α_expected

    Nz = size(grid, 3)
    for k in 2:Nz
        flux_explicit = advective_tracer_flux_z(1, 1, k, grid, scheme, ExplicitTimeDiscretization(), W, c)
        flux_adaptive = advective_tracer_flux_z(1, 1, k, grid, scheme, aiva_td, W, c)
        @test flux_adaptive ≈ expected_scale * flux_explicit rtol=1e-12
    end
end

@testset "AIVA wrapped in WENOVectorInvariant time-steps" begin
    # Regression: the implicit solver path used to access advection.time_discretization
    # as a field, which fails when advection is a VectorInvariant (no such field).
    grid = RectilinearGrid(CPU(), size=(8, 8, 8), x=(0, 1), y=(0, 1), z=(0, 1),
                           halo=(6, 6, 4), topology=(Periodic, Periodic, Bounded))

    momentum_advection = WENOVectorInvariant(; time_discretization=AdaptiveVerticallyImplicitDiscretization(cfl=0.5))
    model = HydrostaticFreeSurfaceModel(grid; momentum_advection, tracer_advection=Centered())

    time_step!(model, 1e-3)
    time_step!(model, 1e-3)

    @test model.clock.iteration == 2
    @test all(isfinite, parent(model.velocities.u))
    @test all(isfinite, parent(model.velocities.v))
end

@testset "AIVA can be re-run after reset!" begin
    grid = RectilinearGrid(CPU(), size=(8, 8, 8), x=(0, 1), y=(0, 1), z=(0, 1),
                           halo=(6, 6, 4), topology=(Periodic, Periodic, Bounded))

    momentum_advection = WENOVectorInvariant(; time_discretization=AdaptiveVerticallyImplicitDiscretization(cfl=0.5))
    model = HydrostaticFreeSurfaceModel(grid; momentum_advection, tracer_advection=Centered(),
                                        timestepper=:SplitRungeKutta3)

    time_step!(model, 1e-3)
    time_step!(model, 1e-3)

    reset!(model.clock)
    @test model.clock.stage == 1

    time_step!(model, 1e-3)

    @test model.clock.iteration == 1
    @test all(isfinite, parent(model.velocities.u))
    @test all(isfinite, parent(model.velocities.v))
end

@testset "AIVA implicit vertical advection of w (z-Face fields)" begin
    Nz = 16
    grid = RectilinearGrid(CPU(), size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, 1000),
                           halo=(1, 1, 4), topology=(Periodic, Periodic, Bounded))

    Δt = 50.0
    td = AdaptiveVerticallyImplicitDiscretization(cfl=0.3)
    scheme = WENO(; time_discretization=td)
    solver = implicit_diffusion_solver(VerticallyImplicitTimeDiscretization(), grid)
    clock = Clock(grid)

    W = ZFaceField(grid)
    set!(W, (x, y, z) -> 5 * exp(-(z - 500)^2 / (2 * 100^2)))   # vanishes in the boundary-adjacent cells
    fill_halo_regions!(W)

    q₀(x, y, z) = exp(-(z - 400)^2 / (2 * 150^2))
    q = ZFaceField(grid)

    Vᶜᶜᶠ(k) = volume(1, 1, k, grid, Center(), Center(), Face())
    column_momentum(q) = sum(Vᶜᶜᶠ(k) * q[1, 1, k] for k in 1:Nz)
    momentum_height(q) = sum(Vᶜᶜᶠ(k) * q[1, 1, k] * znode(1, 1, k, grid, Center(), Center(), Face()) for k in 1:Nz) /
                         column_momentum(q)

    # Explicit limit: below the target CFL the implicit velocity vanishes and the solve is the identity.
    set!(q, q₀)
    fill_halo_regions!(q)
    td.Δt[] = 1e-3
    before = Array(interior(q))
    implicit_step!(q, solver, nothing, nothing, nothing, clock, (;), 1e-3, scheme, (; w=W))
    @test Array(interior(q)) == before

    # Strong splitting: the upwind system conserves the column momentum ∑ Vᶜᶜᶠ w (interior fluxes
    # telescope; wⁱ vanishes at the boundary-adjacent centers), preserves positivity and transports
    # upward, since the advecting velocity is an updraft.
    set!(q, q₀)
    fill_halo_regions!(q)
    td.Δt[] = Δt
    momentum₀ = column_momentum(q)
    height₀ = momentum_height(q)
    implicit_step!(q, solver, nothing, nothing, nothing, clock, (;), Δt, scheme, (; w=W))

    @test all(isfinite, interior(q))
    @test column_momentum(q) ≈ momentum₀
    @test minimum(q[1, 1, k] for k in 1:Nz) ≥ -sqrt(eps(eltype(grid)))
    @test momentum_height(q) > height₀
end

@testset "NonhydrostaticModel steps w at vertical CFL ≫ 1" begin
    Nz = 32
    Δt = 100.0
    grid = RectilinearGrid(CPU(), size=(8, 8, Nz), x=(0, 1000), y=(0, 1000), z=(0, 1000),
                           halo=(4, 4, 4), topology=(Periodic, Periodic, Bounded))

    td = AdaptiveVerticallyImplicitDiscretization(cfl=0.3)
    td.Δt[] = Δt
    model = NonhydrostaticModel(grid; advection=WENO(; time_discretization=td), tracers=:c)

    # α = |w| Δt / Δz ≈ 16, the regime the adaptive split exists for. `w` advects itself, so the
    # implicit coefficients must be evaluated on a `w` the tridiagonal sweep is not overwriting.
    set!(model.velocities.w, (x, y, z) -> 5 * sinpi(2x / 1000) * exp(-(z - 500)^2 / (2 * 150^2)))
    fill_halo_regions!(model.velocities.w)
    set!(model.tracers.c, (x, y, z) -> z / 1000)

    for _ in 1:3
        time_step!(model, Δt)
    end

    @test model.clock.iteration == 3
    @test all(isfinite, interior(model.velocities.w))
    @test maximum(abs, interior(model.velocities.w)) < 10
    @test all(isfinite, interior(model.tracers.c))
end

@testset "Density-weighted implicit advection (mass-flux models)" begin
    grid = RectilinearGrid(CPU(), size=(2, 2, 16), x=(0, 1), y=(0, 1), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded))

    Δt = 50.0
    td = AdaptiveVerticallyImplicitDiscretization(cfl=0.3)
    td.Δt[] = Δt
    scheme = WENO(; time_discretization=td, weight_computation=Oceananigans.Utils.NormalDivision)

    W = ZFaceField(grid)
    set!(W, (x, y, z) -> 5 * sinpi(z / 500))   # exceeds the target CFL over part of the column
    fill_halo_regions!(W)

    ℓx = ℓy = ℓz = Center()

    @testset "ρ ≡ 1 reproduces the volume-conserving coefficients" begin
        ρ = CenterField(grid)
        set!(ρ, 1)
        fill_halo_regions!(ρ)
        for k in 2:15, j in 1:2, i in 1:2
            @test implicit_advection_upper_diagonal(i, j, k, grid, scheme, W, Δt, ℓx, ℓy, ℓz, ρ) ≈
                  implicit_advection_upper_diagonal(i, j, k, grid, scheme, W, Δt, ℓx, ℓy, ℓz)
            @test implicit_advection_lower_diagonal(i, j, k, grid, scheme, W, Δt, ℓx, ℓy, ℓz, ρ) ≈
                  implicit_advection_lower_diagonal(i, j, k, grid, scheme, W, Δt, ℓx, ℓy, ℓz)
            @test implicit_advection_diagonal(i, j, k, grid, scheme, W, Δt, ℓx, ℓy, ℓz, ρ) ≈
                  implicit_advection_diagonal(i, j, k, grid, scheme, W, Δt, ℓx, ℓy, ℓz)
        end
    end

    @testset "off-diagonals scale with the density ratio" begin
        ρ = CenterField(grid)
        set!(ρ, (x, y, z) -> 1 + z / 1000)   # ρ varies smoothly from 1 to 2
        fill_halo_regions!(ρ)
        for k in 2:15, j in 1:2, i in 1:2
            uw = implicit_advection_upper_diagonal(i, j, k, grid, scheme, W, Δt, ℓx, ℓy, ℓz, ρ)
            u0 = implicit_advection_upper_diagonal(i, j, k, grid, scheme, W, Δt, ℓx, ℓy, ℓz)
            @test uw ≈ u0 * ℑzᵃᵃᶠ(i, j, k+1, grid, ρ) / ρ[i, j, k+1]

            lw = implicit_advection_lower_diagonal(i, j, k, grid, scheme, W, Δt, ℓx, ℓy, ℓz, ρ)
            l0 = implicit_advection_lower_diagonal(i, j, k, grid, scheme, W, Δt, ℓx, ℓy, ℓz)
            @test lw ≈ l0 * ℑzᵃᵃᶠ(i, j, k+1, grid, ρ) / ρ[i, j, k]
        end
    end

    @testset "density-weighted implicit solve conserves column mass" begin
        ρ = CenterField(grid)
        set!(ρ, (x, y, z) -> 1 + z / 1000)
        fill_halo_regions!(ρ)

        solver = implicit_diffusion_solver(VerticallyImplicitTimeDiscretization(), grid)
        clock = Clock(grid)

        q = CenterField(grid)
        set!(q, (x, y, z) -> exp(-((z - 500) / 100)^2))
        fill_halo_regions!(q)
        mass₀ = sum(interior(q))

        # No closure ⇒ the solve is the density-weighted implicit vertical advection alone.
        implicit_step!(q, solver, nothing, nothing, Val(1), clock, (;), Δt, scheme, (; w=W), ρ)

        @test all(isfinite, interior(q))
        # The upwind operator is flux-form, so a closed column conserves ∑ V q (uniform V here).
        @test sum(interior(q)) ≈ mass₀ rtol=1e-10
    end
end

@testset "AIVA and bounds preservation both refresh" begin
    grid = RectilinearGrid(CPU(), size=(8, 8, 8), extent=(1, 1, 1), halo=(6, 6, 6))
    advection = WENO(order=5, bounds=(0, 1), time_discretization=AdaptiveVerticallyImplicitDiscretization())
    model = NonhydrostaticModel(grid; advection, tracers=:c)
    scheme = model.advection.c

    @test scheme isa AdaptiveImplicitVerticalAdvection

    set!(model, c=(x, y, z) -> x > 0.5 ? 1.0 : 0.0)
    time_step!(model, 0.25)

    scheme.time_discretization.Δt[] = NaN
    fill!(parent(scheme.bounds.limiter), NaN)

    update_advection!(model.advection, model)

    @test isfinite(scheme.time_discretization.Δt[])
    @test all(isfinite, interior(scheme.bounds.limiter))
end

@testset "AIVA Δt is the Δt of the upcoming RK3 stage" begin
    for arch in archs, FT in float_types
        grid = RectilinearGrid(arch, FT, size=(1, 1, 1), extent=(1, 1, 1))
        ts = RungeKutta3TimeStepper(grid, NamedTuple())
        clock = Clock(grid)
        Δt = 100.0
        Δτ = (ts.γ¹ * Δt, (ts.γ² + ts.ζ²) * Δt, (ts.γ³ + ts.ζ³) * Δt)   # (8/15, 2/15, 1/3) Δt

        # RK3 ticks the clock before `update_state!`: `clock.stage` is the stage about to be taken
        # and `clock.last_stage_Δt` the Δt of the stage that just finished.
        for stage in 1:3
            finished_stage = stage == 1 ? 3 : stage - 1
            clock.stage = stage
            clock.last_Δt = Δt
            clock.last_stage_Δt = Δτ[finished_stage]
            @test adaptive_advection_timestep(ts, clock) ≈ Δτ[stage]
        end

        # First `update_state!` of a run: no stage has finished and `last_stage_Δt` is the first stage's Δt.
        clock.stage = 1
        clock.last_Δt = Δt
        clock.last_stage_Δt = Δτ[1]
        @test adaptive_advection_timestep(ts, clock) ≈ Δτ[1]

        # A Δt that changed between steps is picked up from the finished stage.
        clock.stage = 2
        clock.last_Δt = Δt / 2
        clock.last_stage_Δt = Δτ[1]
        @test adaptive_advection_timestep(ts, clock) ≈ Δτ[2]
    end
end

@testset "AIVA Δt matches the upcoming stage Δt while time stepping" begin
    Δt = 1.0

    # Records (iteration, clock.stage, td.Δt[]) whenever tendencies are computed, i.e. right after
    # `update_advection!` has refreshed td.Δt[] for the substep about to be taken.
    function recorded_advection_timesteps(model)
        td = time_discretization(model.advection.c)
        seen = Tuple{Int, Int, Float64}[]
        record(model) = push!(seen, (model.clock.iteration, model.clock.stage, td.Δt[]))
        callbacks = [Callback(record, callsite=TendencyCallsite())]
        time_step!(model, Δt; callbacks)
        time_step!(model, Δt; callbacks)
        return seen
    end

    for arch in archs, FT in float_types
        A = typeof(arch)
        grid = RectilinearGrid(arch, FT, size=(4, 4, 4), extent=(1, 1, 1))
        advection = WENO(FT; time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.3))
        rtol = 10 * eps(FT)

        @testset "RungeKutta3TimeStepper [$A, $FT]" begin
            model = NonhydrostaticModel(grid; advection, tracers=:c, timestepper=:RungeKutta3)
            ts = model.timestepper
            Δτ = (ts.γ¹ * Δt, (ts.γ² + ts.ζ²) * Δt, (ts.γ³ + ts.ζ³) * Δt)
            seen = recorded_advection_timesteps(model)
            @test length(seen) == 1 + 2 * 3
            # The clock is ticked before `update_state!`, so `clock.stage` is the stage about to be taken.
            for (iteration, stage, advection_timestep) in seen
                @test advection_timestep ≈ Δτ[stage] rtol=rtol
            end
        end

        @testset "QuasiAdamsBashforth2TimeStepper [$A, $FT]" begin
            model = NonhydrostaticModel(grid; advection, tracers=:c, timestepper=:QuasiAdamsBashforth2)
            seen = recorded_advection_timesteps(model)
            @test length(seen) == 1 + 2
            for (iteration, stage, advection_timestep) in seen
                @test advection_timestep ≈ Δt rtol=rtol
            end
        end

        @testset "SplitRungeKuttaTimeStepper [$A, $FT]" begin
            model = HydrostaticFreeSurfaceModel(grid; tracer_advection=advection, tracers=:c, timestepper=:SplitRungeKutta3)
            β = model.timestepper.β
            Nstages = model.timestepper.Nstages
            seen = recorded_advection_timesteps(model)
            @test length(seen) == 1 + 2 * Nstages
            # `clock.stage` is set before each substep, so at the callback it is the stage that just finished.
            # The first `update_state!` of the run precedes every stage but looks like stage 1 having finished,
            # so its td.Δt[] is that of stage 2 rather than stage 1; it is skipped here.
            for (iteration, stage, advection_timestep) in seen[2:end]
                next_stage = stage == Nstages ? 1 : stage + 1
                @test advection_timestep ≈ Δt / β[next_stage] rtol=rtol
            end
        end
    end
end

@testset "AIVA reproduces the explicit scheme below the stage CFL threshold" begin
    # A cellular flow from a discrete streamfunction is exactly divergence-free, so the pressure projection
    # leaves it untouched and the vertical CFL of every stage is known exactly.
    Nx, Nz = 16, 32
    Δx, Δz = 1 / Nx, 1 / Nz
    ψ(x, z) = sinpi(2x) * sinpi(z) / (2π)                     # ∂ψ/∂x = cospi(2x) sinpi(z), so max |w| ≈ 1
    Ψ = [ψ(i * Δx, k * Δz) for i in 0:Nx, k in 0:Nz]           # cell corners; Ψ[Nx+1, :] wraps periodically
    u = [-(Ψ[i, k+1] - Ψ[i, k]) / Δz for i in 1:Nx, _ in 1:1, k in 1:Nz]
    w = [ (Ψ[i+1, k] - Ψ[i, k]) / Δx for i in 1:Nx, _ in 1:1, k in 1:Nz+1]
    wmax = maximum(abs, w)

    # The longest RK3 stage is the first, γ¹ Δt, so the split is inert in every stage while γ¹ |w| Δt / Δz ≤ cfl.
    cfl = 0.5
    γ¹ = 8 / 15
    explicit_limit = cfl * Δz / (γ¹ * wmax)
    nsteps = 10

    for arch in archs, FT in float_types
        A = typeof(arch)
        grid = RectilinearGrid(arch, FT, size=(Nx, Nz), x=(0, 1), z=(0, 1), topology=(Periodic, Flat, Bounded))

        function final_tracer(time_discretization, Δt)
            tracer_advection = WENO(FT; time_discretization)
            model = NonhydrostaticModel(grid; momentum_advection=nothing, tracer_advection, tracers=:c,
                                        timestepper=:RungeKutta3)
            set!(model; u, w, c=(x, z) -> sinpi(2x))
            for _ in 1:nsteps
                time_step!(model, Δt)
            end
            return Array(interior(model.tracers.c))
        end

        @testset "AIVA equals explicit scheme below threshold [$A, $FT]" begin
            Δt = 0.95 * explicit_limit
            explicit_tracer = final_tracer(ExplicitTimeDiscretization(), Δt)
            adaptive_tracer = final_tracer(AdaptiveVerticallyImplicitDiscretization(FT; cfl), Δt)
            @test maximum(abs, adaptive_tracer .- explicit_tracer) < 100 * eps(FT)

            # Just above the threshold the first stage is partly implicit and the two schemes differ.
            Δt = 1.05 * explicit_limit
            explicit_tracer = final_tracer(ExplicitTimeDiscretization(), Δt)
            adaptive_tracer = final_tracer(AdaptiveVerticallyImplicitDiscretization(FT; cfl), Δt)
            @test maximum(abs, adaptive_tracer .- explicit_tracer) > 1e-4
        end
    end
end
