include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, mask_immersed_field!
using Oceananigans.Advection:
        _symmetric_interpolate_xᶠᵃᵃ,
        _symmetric_interpolate_xᶜᵃᵃ,
        _symmetric_interpolate_yᵃᶠᵃ,
        _symmetric_interpolate_yᵃᶜᵃ,
        _biased_interpolate_xᶜᵃᵃ,
        _biased_interpolate_xᶠᵃᵃ,
        _biased_interpolate_yᵃᶜᵃ,
        _biased_interpolate_yᵃᶠᵃ,
        FluxFormAdvection,
        LeftBias,
        RightBias,
        materialize_advection

using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.TimeSteppers: compute_tendencies!

linear_advection_schemes = [Centered, UpwindBiased]
advection_schemes = [linear_advection_schemes... WENO]

@inline advective_order(buffer, ::Type{Centered}) = buffer * 2
@inline advective_order(buffer, AdvectionType)    = buffer * 2 - 1

function run_tracer_interpolation_test(c, ibg, scheme)
    scheme = materialize_advection(scheme, ibg)
    for j in 6:19, i in 6:19
        if typeof(scheme) <: Centered
            @test @allowscalar  _symmetric_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, scheme, c) ≈ 1.0
        else
            @test @allowscalar _biased_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, scheme, LeftBias,  c) ≈ 1.0
            @test @allowscalar _biased_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, scheme, RightBias, c) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶠᵃ(i, j+1, 1, ibg, scheme, LeftBias,  c) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶠᵃ(i, j+1, 1, ibg, scheme, RightBias, c) ≈ 1.0
        end
    end
end

function run_tracer_conservation_test(grid, scheme)
    scheme = materialize_advection(scheme, grid)
    model = HydrostaticFreeSurfaceModel(grid; tracers = :c,
                                        free_surface = ExplicitFreeSurface(),
                                        tracer_advection = scheme)

    c = model.tracers.c
    set!(model, c = 1)
    fill_halo_regions!(c)

    η = model.free_surface.displacement

    indices = model.grid isa ImmersedBoundaryGrid ? (5:7, 3:6, 1) : (2:5, 3:6, 1)

    interior(η, indices...) .= - 0.05
    fill_halo_regions!(η)

    wave_speed = sqrt(model.free_surface.gravitational_acceleration)
    dt = 0.1 / wave_speed
    for _ in 1:10
        time_step!(model, dt)
    end

    @test maximum(c) ≈ 1.0
    @test minimum(c) ≈ 1.0
    @test mean(c)    ≈ 1.0

    return nothing
end

function run_momentum_interpolation_test(u, v, ibg, scheme)
    scheme = materialize_advection(scheme, ibg)

    # ensure also immersed boundaries have a value of 1
    interior(u, 6, :, 1) .= 1.0
    interior(v, :, 6, 1) .= 1.0

    for i in 7:19, j in 7:19
        if typeof(scheme) <: Centered
            @test @allowscalar  _symmetric_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, u) ≈ 1.0
            @test @allowscalar  _symmetric_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, v) ≈ 1.0
            @test @allowscalar  _symmetric_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, u) ≈ 1.0
            @test @allowscalar  _symmetric_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, v) ≈ 1.0
        else
            @test @allowscalar _biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, LeftBias,  u) ≈ 1.0
            @test @allowscalar _biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, RightBias, u) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, LeftBias,  u) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, RightBias, u) ≈ 1.0

            @test @allowscalar _biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, LeftBias,  v) ≈ 1.0
            @test @allowscalar _biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, RightBias, v) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, LeftBias,  v) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, RightBias, v) ≈ 1.0
        end
    end

    return nothing
end

staircase(ξ) = (5 + sum(tanh(40 * (ξ - n / 6)) for n in 1:5)) / 20

function run_kinetic_energy_conservation_test(arch)
    grid = RectilinearGrid(arch, size=(16, 16, 32), x=(0, 1), y=(0, 1), z=(0, 1), topology=(Bounded, Bounded, Bounded))
    ibg  = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> staircase(x) + staircase(y)))

    pressure_solver = ConjugateGradientPoissonSolver(ibg; reltol=1e-14, abstol=0, maxiter=1000)
    model = NonhydrostaticModel(ibg; advection=Centered(), pressure_solver)

    u, v, w = model.velocities
    set!(model, u = rand(size(u)...) .- 1/2, v = rand(size(v)...) .- 1/2, w = rand(size(w)...) .- 1/2)
    compute_tendencies!(model, [])
    G = model.timestepper.Gⁿ

    production = sum(u * G.u) + sum(v * G.v) + sum(w * G.w)
    magnitude  = sum(abs, u * G.u) + sum(abs, v * G.v) + sum(abs, w * G.w)

    @test abs(production) < 1e-12 * magnitude

    return nothing
end

function run_staircase_convection_test(arch)
    grid = RectilinearGrid(arch, size=(16, 128), x=(0, 1), z=(0, 1), topology=(Bounded, Flat, Bounded))
    ibg  = ImmersedBoundaryGrid(grid, PartialCellBottom(x -> 2staircase(x)))

    model = NonhydrostaticModel(ibg; advection=WENO(), tracers=:b, buoyancy=BuoyancyTracer(),
                                pressure_solver=ConjugateGradientPoissonSolver(ibg))

    set!(model, b = (x, z) -> - exp(-((x - 1/2)^2 + (z - 0.55)^2) / (2 * 0.05^2)))

    simulation = Simulation(model; Δt=1e-3, stop_time=1, verbose=false)
    conjure_time_step_wizard!(simulation, cfl=0.7)
    run!(simulation)

    @test maximum(abs, model.velocities.u) < 1

    return nothing
end

for arch in archs
    @testset "Immersed tracer reconstruction" begin
        @info "Running immersed tracer reconstruction tests..."

        grid = RectilinearGrid(arch, size=(20, 20), extent=(20, 20), halo = (6, 6), topology=(Bounded, Bounded, Flat))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y) -> (x < 5 || y < 5)))

        c = CenterField(ibg)
        set!(c, 1)
        mask_immersed_field!(c)
        fill_halo_regions!(c)

        for adv in linear_advection_schemes, buffer in [1, 2, 3, 4, 5]
            scheme = adv(order = advective_order(buffer, adv))

            @info "  Testing immersed tracer reconstruction [$(typeof(arch)), $(summary(scheme))]"
            run_tracer_interpolation_test(c, ibg, scheme)
        end

        for buffer in [2, 3, 4, 5], bounds in (nothing, (0, 1))
            scheme = WENO(; order = advective_order(buffer, WENO), bounds)

            @info "  Testing immersed tracer reconstruction [$(typeof(arch)), $(summary(scheme))]"
            run_tracer_interpolation_test(c, ibg, scheme)
        end
    end

    @testset "Immersed tracer conservation" begin
        @info "Running immersed tracer conservation tests..."

        grid = RectilinearGrid(arch, size=(10, 8, 1), extent=(10, 8, 1), halo = (6, 6, 6), topology=(Bounded, Periodic, Bounded))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> ifelse(x < 2, 0, -1)))

        for adv in advection_schemes, buffer in [1, 2, 3, 4, 5]
            scheme = adv(order = advective_order(buffer, adv))

            for g in [grid, ibg]
                @info "  Testing immersed tracer conservation [$(typeof(arch)), $(summary(scheme)), $(typeof(g).name.wrapper)]"
                run_tracer_conservation_test(g, scheme)
            end
        end

        for adv in advection_schemes, buffer in [1, 2, 3, 4, 5]
            directional_scheme = adv(order = advective_order(buffer, adv))
            scheme = FluxFormAdvection(directional_scheme, directional_scheme, directional_scheme)
            for g in [grid, ibg]
                @info "  Testing immersed tracer conservation [$(typeof(arch)), $(summary(scheme)), $(typeof(g).name.wrapper)]"
                run_tracer_conservation_test(g, scheme)
            end
        end
    end

    @testset "Immersed momentum reconstruction" begin
        @info "Running immersed momentum reconstruction tests..."

        grid = RectilinearGrid(arch, size=(20, 20), extent=(20, 20), halo = (6, 6), topology=(Bounded, Bounded, Flat))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y) -> (x < 5 || y < 5)))

        u = XFaceField(ibg)
        v = YFaceField(ibg)
        set!(u, 1)
        set!(v, 1)

        mask_immersed_field!(u)
        mask_immersed_field!(v)

        fill_halo_regions!(u)
        fill_halo_regions!(v)

        for adv in advection_schemes, buffer in [1, 2, 3, 4, 5]
            scheme = adv(order = advective_order(buffer, adv))

            @info "  Testing immersed momentum reconstruction [$(typeof(arch)), $(summary(scheme))]"
            run_momentum_interpolation_test(u, v, ibg, scheme)
        end
    end

    @testset "Immersed momentum advection conserves kinetic energy" begin
        @info "Running immersed momentum advection kinetic energy test [$(typeof(arch))]..."
        run_kinetic_energy_conservation_test(arch)
    end

    @testset "Convection over an immersed staircase stays bounded" begin
        @info "Running immersed staircase convection test [$(typeof(arch))]..."
        run_staircase_convection_test(arch)
    end
end
