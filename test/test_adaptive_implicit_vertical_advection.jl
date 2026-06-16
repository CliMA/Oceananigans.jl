include("dependencies_for_runtests.jl")

using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection,
                              advective_tracer_flux_z,
                              needs_implicit_solver,
                              update_advection_timestep!
using Oceananigans.TimeSteppers: AdaptiveVerticallyImplicitDiscretization,
                                 ExplicitTimeDiscretization,
                                 adaptive_implicit_vertical_advection_diagnostics,
                                 time_discretization

@testset "AdaptiveVerticallyImplicitDiscretization construction" begin
    td = AdaptiveVerticallyImplicitDiscretization(maximum_explicit_cfl=0.3)
    @test td.maximum_explicit_cfl == 0.3
    @test isnothing(td.implicit_fraction)
    @test td.cfl[] == 0.3
    @test td.Δt[] == zero(td.cfl[])

    td_alias = AdaptiveVerticallyImplicitDiscretization(cfl=0.25)
    @test td_alias.maximum_explicit_cfl == 0.25
    @test td_alias.cfl[] == 0.25

    td_fraction = AdaptiveVerticallyImplicitDiscretization(implicit_fraction=0.4)
    @test isnothing(td_fraction.maximum_explicit_cfl)
    @test td_fraction.implicit_fraction == 0.4
    @test td_fraction.sample_top_levels == 10
    @test td_fraction.sample_bottom_levels == 10
    @test td_fraction.cfl[] == 0.0

    td_boundary_sample = AdaptiveVerticallyImplicitDiscretization(implicit_fraction=0.4, sample_top_levels=3, sample_bottom_levels=2)
    @test td_boundary_sample.sample_top_levels == 3
    @test td_boundary_sample.sample_bottom_levels == 2

    td_f32 = AdaptiveVerticallyImplicitDiscretization(Float32; implicit_fraction=0.4)
    @test td_f32.implicit_fraction isa Float32
    @test td_f32.implicit_fraction == 0.4f0

    @test_throws ArgumentError AdaptiveVerticallyImplicitDiscretization()
    @test_throws ArgumentError AdaptiveVerticallyImplicitDiscretization(cfl=0.3, maximum_explicit_cfl=0.3)
    @test_throws ArgumentError AdaptiveVerticallyImplicitDiscretization(maximum_explicit_cfl=0.3, implicit_fraction=0.4)
    @test_throws ArgumentError AdaptiveVerticallyImplicitDiscretization(implicit_fraction=-0.1)
    @test_throws ArgumentError AdaptiveVerticallyImplicitDiscretization(implicit_fraction=1.1)
    @test_throws ArgumentError AdaptiveVerticallyImplicitDiscretization(implicit_fraction=0.4, sample_top_levels=-1)
    @test_throws ArgumentError AdaptiveVerticallyImplicitDiscretization(implicit_fraction=0.4, sample_bottom_levels=-1)
end

@testset "AIVA scheme dispatch" begin
    for scheme in (WENO, Centered, UpwindBiased)
        extra_kw = scheme == WENO ? (; weight_computation = Oceananigans.Utils.NormalDivision) : (; )
        explicit = scheme(; extra_kw...)
        adaptive = scheme(; time_discretization = AdaptiveVerticallyImplicitDiscretization(maximum_explicit_cfl=0.5), extra_kw...)

        @test !(explicit isa AdaptiveImplicitVerticalAdvection)
        @test adaptive  isa AdaptiveImplicitVerticalAdvection

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
    aiva_td = AdaptiveVerticallyImplicitDiscretization(maximum_explicit_cfl=10.0)
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
    aiva_td = AdaptiveVerticallyImplicitDiscretization(maximum_explicit_cfl=cfl)
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

function build_nonhydrostatic_aiva_model(td; grid_size=(4, 4, 8), z_extent=(0, grid_size[3]))
    grid = RectilinearGrid(CPU(), size=grid_size, x=(0, 1), y=(0, 1), z=z_extent,
                           halo=(6, 6, 4), topology=(Periodic, Periodic, Bounded))

    tracer_advection = WENO(; time_discretization=td, weight_computation=Oceananigans.Utils.NormalDivision)
    model = NonhydrostaticModel(grid; tracers=:c, tracer_advection, buoyancy=nothing, closure=nothing)

    set!(model, w=(x, y, z) -> z)
    fill_halo_regions!(model.velocities.w)
    return model
end

@testset "AIVA implicit_fraction resolves the expected boundary-sampled percentile threshold" begin
    td = AdaptiveVerticallyImplicitDiscretization(implicit_fraction=0.25, sample_top_levels=2, sample_bottom_levels=2)
    model = build_nonhydrostatic_aiva_model(td; grid_size=(4, 4, 20), z_extent=(0, 20))
    model.clock.last_Δt = 1.0

    update_advection_timestep!(model.advection, model.timestepper, model.clock, model)
    diagnostics = adaptive_implicit_vertical_advection_diagnostics(td)

    @test diagnostics.resolved_cfl ≈ 18.75
    @test diagnostics.realized_implicit_fraction ≈ 0.25
    @test diagnostics.median_cfl ≈ 10.0
    @test diagnostics.max_cfl ≈ 19.5
end

@testset "AIVA boundary sampling deduplicates shallow columns" begin
    td = AdaptiveVerticallyImplicitDiscretization(implicit_fraction=0.5, sample_top_levels=10, sample_bottom_levels=10)
    model = build_nonhydrostatic_aiva_model(td; grid_size=(4, 4, 8), z_extent=(0, 8))
    model.clock.last_Δt = 1.0

    update_advection_timestep!(model.advection, model.timestepper, model.clock, model)
    diagnostics = adaptive_implicit_vertical_advection_diagnostics(td)

    @test diagnostics.resolved_cfl ≈ 4.0
    @test diagnostics.realized_implicit_fraction ≈ 0.5
    @test diagnostics.median_cfl ≈ 4.0
    @test diagnostics.max_cfl ≈ 7.5
end

@testset "AIVA implicit_fraction = 0 reproduces explicit vertical advection" begin
    grid = RectilinearGrid(CPU(), size=(1, 1, 8),
                           x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Bounded))

    Δt = 1.0
    aiva_td = AdaptiveVerticallyImplicitDiscretization(implicit_fraction=0.0)
    aiva_td.Δt[] = Δt
    aiva_td.cfl[] = abs(1.0) * Δt / (1 / size(grid, 3))

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

@testset "AIVA wrapped in WENOVectorInvariant time-steps" begin
    # Regression: the implicit solver path used to access advection.time_discretization
    # as a field, which fails when advection is a VectorInvariant (no such field).
    grid = RectilinearGrid(CPU(), size=(8, 8, 8), x=(0, 1), y=(0, 1), z=(0, 1),
                           halo=(6, 6, 4), topology=(Periodic, Periodic, Bounded))

    momentum_advection = WENOVectorInvariant(; time_discretization=AdaptiveVerticallyImplicitDiscretization(maximum_explicit_cfl=0.5))
    model = HydrostaticFreeSurfaceModel(grid; momentum_advection, tracer_advection=Centered())

    time_step!(model, 1e-3)
    time_step!(model, 1e-3)

    @test model.clock.iteration == 2
    @test all(isfinite, parent(model.velocities.u))
    @test all(isfinite, parent(model.velocities.v))
end
