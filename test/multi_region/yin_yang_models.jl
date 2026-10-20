include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "yin_yang_test_utils.jl"))

using Oceananigans: fields
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: FixedSubstepNumber
using Oceananigans.TurbulenceClosures: implicit_step!

geographic_velocity(component) = (λ, φ, z) -> solid_body_velocity(λ, φ)[component]

# A free-surface bump in the overlap at (λ, φ) = (-90°, 0), which lies in both panels
bump(λ, φ, z) = exp(-((λ + 90)^2 + φ^2) / 400)

hydrostatic_model(grid; kwargs...) = HydrostaticFreeSurfaceModel(grid; momentum_advection = VectorInvariant(),
                                                                 buoyancy = nothing, closure = nothing, kwargs...)

split_explicit_model(grid; kwargs...) = hydrostatic_model(grid; tracer_advection = nothing, kwargs...,
                                                          free_surface = SplitExplicitFreeSurface(substeps=10, extend_halos=false))

# The largest difference between the z-star scaling σ = (H + η) / H at cell centers and its value from η
zstar_scaling_error(model, H) = maximum(1:2) do r
    panel = getregion(model.grid, r)
    σ = on_architecture(CPU(), panel.z.σᶜᶜⁿ)[1:size(panel, 1), 1:size(panel, 2), 1]
    η = Array(interior(getregion(model.free_surface.displacement, r), :, :, 1))
    return maximum(abs, σ .- (H .+ η) ./ H)
end

@testset "Yin-Yang models [$(summary(arch))]" for arch in archs
    grid = YinYangGrid(arch; size=(48, 16, 2), z=(-1000, 0))
    Nx, Ny, Nz = size(grid)
    Δt = 600

    @testset "set! and the split-explicit free surface" begin
        boundary_conditions = (; c = FieldBoundaryConditions(top = ValueBoundaryCondition((λ, φ, t) -> 2t)),
                                 d = FieldBoundaryConditions(top = ValueBoundaryCondition((λ, φ, t, b) -> b,
                                                                                        field_dependencies = :b)))
        model = split_explicit_model(grid; tracers = (:b, :c, :d), boundary_conditions)
        u, v, _ = model.velocities

        # set! takes geographic velocities and rotates them into each panel's basis
        set!(model, u = geographic_velocity(1), v = geographic_velocity(2))
        for (field, component) in ((u, panel_u), (v, panel_v))
            @test panel_error(field, component(solid_body_velocity, grid); nodes = interior_nodes) < 1e-12
            @test panel_error(field, component(solid_body_velocity, grid)) < 2e-5
        end

        # A missing component is zero, setting a tracer leaves the velocities untouched, panel components are set as they are
        zonal_velocity(λ, φ) = (cosd(φ), zero(φ))
        set!(model, u = (λ, φ, z) -> cosd(φ))
        set!(model, b = 5)
        @test panel_error(u, panel_u(zonal_velocity, grid); nodes = interior_nodes) < 1e-12
        @test panel_error(v, panel_v(zonal_velocity, grid); nodes = interior_nodes) < 1e-12
        set!(model, u = 1, intrinsic_velocities = true)
        @test all(==(1), interior(getregion(u, 2)))

        # Without tracer advection, the top halos are set by time- and field-dependent conditions
        set!(model, u = 0, v = 0, η = bump, c = 1, d = 1)
        for _ in 1:2
            time_step!(model, Δt)
        end
        t = model.clock.time
        for r in 1:2
            @test all(view(getregion(model.tracers.c, r).data, 1:Nx, 1:Ny, Nz+1) .≈ 2 * 2t - 1)
            @test all(view(getregion(model.tracers.d, r).data, 1:Nx, 1:Ny, Nz+1) .≈ 2 * 5 - 1)
        end

        # SplitRungeKutta3 gives the same wave
        rk3_model = split_explicit_model(grid; timestepper = :SplitRungeKutta3)
        set!(rk3_model, η = bump)
        for _ in 1:2
            time_step!(rk3_model, Δt)
        end
        η, ηᴬᴮ = rk3_model.free_surface.displacement, model.free_surface.displacement
        @test 0.3 < maximum(interior(getregion(η, 1))) ≤ 1
        for r in 1:2
            @test interior(getregion(η, r)) ≈ interior(getregion(ηᴬᴮ, r)) rtol=1e-5
        end

        @test_throws ArgumentError hydrostatic_model(grid; free_surface = SplitExplicitFreeSurface(substeps=10))
    end

    mountain(λ, φ) = -1000 + 400 * exp(-(λ^2 + (φ - 30)^2) / 200)
    zstar_grid = YinYangGrid(arch; size=(48, 16, 2), z=MutableVerticalDiscretization((-1000, 0)), halo=(4, 4, 4))
    immersed_zstar_grid = ImmersedBoundaryGrid(zstar_grid, PartialCellBottom(mountain))

    @testset "Halos are consistent after a time step [$timestepper]" for (timestepper, step_grid) in
            ((:QuasiAdamsBashforth2, grid), (:SplitRungeKutta3, immersed_zstar_grid))

        model = hydrostatic_model(step_grid; free_surface = ExplicitFreeSurface(), tracer_advection = WENO(order=5),
                                  coriolis = HydrostaticSphericalCoriolis(), tracers = :c, timestepper)
        set!(model, u = geographic_velocity(1), v = geographic_velocity(2), η = bump, c = (λ, φ, z) -> bump(λ + 60, φ, z))
        V₀ = active_volume(model)
        callbacks = conservation_fixer_callbacks(model)
        for _ in 1:2
            time_step!(model, 300; callbacks)
        end
        @test active_volume(model) ≈ V₀ rtol = 1e-14

        # A refill right after `time_step!` changes nothing: no kernel writes to the halos after the step's fills
        u, v, _ = model.velocities
        c = model.tracers.c
        η = model.free_surface.displacement
        before = map(regional_parents, (u, v, c, η))
        fill_halo_regions!((u, v), model.clock, fields(model))
        fill_halo_regions!((c, η), model.clock, fields(model))
        @test map(regional_parents, (u, v, c, η)) == before

        # The halo columns get their vertical halos from the donor's, which the no-flux conditions set
        halos = halo_nodes(step_grid)
        for field in (u, v, c), r in 1:2
            data = on_architecture(CPU(), getregion(field, r).data)
            @test [data[i, j, 0] for (i, j) in halos] == [data[i, j, 1] for (i, j) in halos]
            @test [data[i, j, Nz+1] for (i, j) in halos] == [data[i, j, Nz] for (i, j) in halos]
        end
    end

    # The immersed boundaries of the panels move to another architecture as they are
    for r in 1:2
        panel = getregion(immersed_zstar_grid, r)
        cpu_panel = on_architecture(CPU(), panel)
        @test cpu_panel isa ImmersedBoundaryGrid
        @test parent(cpu_panel.immersed_boundary.bottom_height) == Array(parent(panel.immersed_boundary.bottom_height))
    end

    # The default free surface substeps without extending the halos
    default_model = hydrostatic_model(immersed_zstar_grid; tracer_advection = nothing)
    @test default_model.free_surface isa SplitExplicitFreeSurface
    @test default_model.free_surface.substepping isa FixedSubstepNumber

    @testset "z-star on a partitioned latitude-longitude grid" begin
        H = 1000
        lat_lon_grid = LatitudeLongitudeGrid(arch; size=(16, 8, 1), z=MutableVerticalDiscretization((-H, 0)),
                                             longitude=(-180, 180), latitude=(-80, 80), radius=1e6)
        model = hydrostatic_model(MultiRegionGrid(lat_lon_grid, partition = XPartition(2));
                                  free_surface = ExplicitFreeSurface(), tracers = (), timestepper = :SplitRungeKutta3)
        set!(model, η = (λ, φ, z) -> 100 * exp(-(λ^2 + φ^2) / 400))
        @test zstar_scaling_error(model, H) < 1e-12
        time_step!(model, 1)
        @test zstar_scaling_error(model, H) < 1e-12
    end
end

@testset "Implicit diffusion on Yin-Yang grids [$(summary(arch)), immersed = $immersed]" for arch in archs, immersed in (false, true)
    sphere = YinYangGrid(arch, Float32; size=(48, 16, 4), z=(-4, 0))
    grid = immersed ? ImmersedBoundaryGrid(sphere, GridFittedBottom((λ, φ) -> -4)) : sphere
    closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), Float32; κ=1)
    model = HydrostaticFreeSurfaceModel(grid; closure, tracers=:c, buoyancy=nothing,
                                        momentum_advection=nothing, tracer_advection=nothing)
    set!(model, c=(λ, φ, z) -> 1 + cospi((z + 4) / 4))
    damping = 1 / (1 + 4sinpi(1/8)^2) # of the cosine by one backward Euler step with Δt = 1
    expected = CenterField(grid)
    set!(expected, (λ, φ, z) -> 1 + damping * cospi((z + 4) / 4))

    for region in 1:2
        m = getregion(model, region)
        implicit_step!(m.tracers.c, m.timestepper.implicit_solver, m.closure, m.closure_fields,
                       Val(1), m.clock, fields(m), 1f0)
        actual = Array(interior(m.tracers.c))
        @test actual ≈ Array(interior(getregion(expected, region))) rtol=5f-6
        @test mean(Float64, actual) ≈ 1 rtol=5f-6
    end
end

@testset "Bounded advection on Yin-Yang grids [$(summary(arch))]" for arch in archs
    grid = YinYangGrid(arch; size=(48, 16, 4), halo=(4, 4, 3), z=(-1000, 0))
    velocities = PrescribedVelocityFields(u=XFaceField(grid; boundary_conditions=nothing),
                                         v=YFaceField(grid; boundary_conditions=nothing),
                                         w=ZFaceField(grid; boundary_conditions=nothing))
    model = HydrostaticFreeSurfaceModel(grid; velocities, momentum_advection=nothing, tracers=:c, buoyancy=nothing,
                                        tracer_advection=WENO(order=5, bounds=(0, 1)), timestepper=:SplitRungeKutta3)
    set!(model, c=(λ, φ, z) -> ifelse(cosd(λ) * cosd(φ) > 1/2, 1, 0))
    time_step!(model, 1)
    limiter = model.advection.c.bounds.limiter
    @test 0 ≤ minimum(limiter) < 1
    @test maximum(limiter) ≤ 1
    @test minimum(model.tracers.c) == 0
    @test maximum(model.tracers.c) == 1
end
