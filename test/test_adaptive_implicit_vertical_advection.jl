include("dependencies_for_runtests.jl")

using Oceananigans.BoundaryConditions: needs_implicit_solver

using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection,
                              advective_tracer_flux_z

using Oceananigans.TimeSteppers: AdaptiveVerticallyImplicitDiscretization,
                                 ExplicitTimeDiscretization,
                                 time_discretization

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
