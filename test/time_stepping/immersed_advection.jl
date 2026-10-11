include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, mask_immersed_field!
using Oceananigans.Utils: NormalDivision
using Oceananigans.Advection:
        GhostCells,
        ghost_cell_reconstruction,
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

    @testset "GhostCells boundary reconstruction" begin
        @info "Running GhostCells boundary reconstruction tests..."

        scheme = WENO(order=7, weight_computation=NormalDivision, boundary_scheme=GhostCells()).buffer_scheme

        function reconstruct(scheme, bias, values, activity)
            upwind_values   = bias == LeftBias ? values   : reverse(values)
            upwind_activity = bias == LeftBias ? activity : reverse(activity)
            upstream_run    = accumulate(&, upwind_activity[3:-1:1])
            downstream_run  = accumulate(&, upwind_activity[5:7])
            return ghost_cell_reconstruction(scheme, bias, upwind_values[1:7], upstream_run, downstream_run, nothing)
        end

        # Averages over the unit cells [n - 1, n] of a polynomial with primitive P; the reconstructed face is at x = 4
        averages(P) = ntuple(n -> P(n) - P(n - 1), 8)

        active   = ntuple(_ -> true, 8)
        wall     = (false, false, true, true, true, true, true, true)
        two_left = (false, false, false, true, true, false, false, false)

        quadratic_primitive(x) = x^3 / 3 - 3x^2 / 2 + x   # p(x) = x² - 3x + 1
        linear_primitive(x)    = x^2 / 2 + 2x             # p(x) = x + 2

        for bias in (LeftBias, RightBias)
            # the values in inactive cells never enter the reconstruction
            @test reconstruct(scheme, bias, (1e3, 1e3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), wall) == 3.0
            @test reconstruct(scheme, bias, Tuple(1.0:8.0), active) == 4.5

            # the ghosts extrapolate quadratically, so quadratic data are reconstructed exactly next to the wall
            @test reconstruct(scheme, bias, averages(quadratic_primitive), wall) ≈ 4^2 - 3 * 4 + 1

            # with only two active cells the extrapolation is linear
            @test reconstruct(scheme, bias, averages(linear_primitive), two_left) ≈ 4 + 2
        end

        grid = RectilinearGrid(arch, size=(20, 20), extent=(20, 20), halo=(6, 6), topology=(Bounded, Bounded, Flat))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y) -> (x < 5 || y < 5)))

        c = CenterField(ibg)
        set!(c, 1)
        mask_immersed_field!(c, 1e3)
        fill_halo_regions!(c)

        weno = materialize_advection(WENO(order=7, boundary_scheme=GhostCells()), ibg)
        for j in 6:19, i in 6:19, bias in (LeftBias, RightBias)
            @test @allowscalar _biased_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, weno, bias, c) ≈ 1.0
            @test @allowscalar _biased_interpolate_yᵃᶠᵃ(i, j+1, 1, ibg, weno, bias, c) ≈ 1.0
        end
    end

    @testset "GhostCells projection" begin
        @info "Running GhostCells projection tests..."

        Nx, Nz = 64, 32
        grid = RectilinearGrid(arch, size=(Nx, Nz), x=(0, 1), z=(-1, 0), halo=(6, 6), topology=(Periodic, Flat, Bounded))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBottom(x -> 0.35 < x < 0.65 ? -0.35 : -1.0))

        # A divergence-free velocity from a streamfunction that vanishes on the staircase
        ψ(x, z) = 0.05 * sin(2π * x) * sin(π * (z + 1))
        dry(i, k) = k < 1 || k > Nz || (0.35 < (mod1(i, Nx) - 0.5) / Nx < 0.65 && -1 + (k - 0.5) / Nz < -0.35)
        Ψ = [(dry(i-1, k-1) | dry(i, k-1) | dry(i-1, k) | dry(i, k)) ? 0.0 : ψ((i - 1) / Nx, -1 + (k - 1) / Nz) for i in 1:Nx+1, k in 1:Nz+1]
        u = XFaceField(ibg)
        w = ZFaceField(ibg)
        set!(u, [-(Ψ[i, k+1] - Ψ[i, k]) * Nz for i in 1:Nx, _ in 1:1, k in 1:Nz])
        set!(w, [ (Ψ[i+1, k] - Ψ[i, k]) * Nx for i in 1:Nx, _ in 1:1, k in 1:Nz+1])

        initial_conditions = (c = (x, z) -> ifelse(-0.6 < z < -0.4, 1.0, 0.0),
                              d = (x, z) -> ifelse(-0.9 < z < -0.7, 1.0, 0.0))

        function advected_tracers(scheme, tracers = (:c,))
            model = HydrostaticFreeSurfaceModel(ibg; velocities = PrescribedVelocityFields(; u, w), tracers,
                                                tracer_advection = scheme, timestepper = :SplitRungeKutta3)
            set!(model; NamedTuple{tracers}(map(name -> initial_conditions[name], tracers))...)
            c = model.tracers[1]
            total₀ = sum(interior(c))
            cmin, cmax = 0.0, 1.0
            for _ in 1:100
                time_step!(model, 0.01)
                cmin = min(cmin, minimum(interior(c)))
                cmax = max(cmax, maximum(interior(c)))
            end
            return (; model, cmin, cmax, drift = sum(interior(c)) - total₀)
        end

        bounded   = advected_tracers(WENO(order=7, boundary_scheme=GhostCells()))
        unbounded = advected_tracers(WENO(order=7, boundary_scheme=GhostCells(bounded=false)))

        # The projection removes the extrema generated next to the boundary, leaving the quasi-bounds of the
        # interior reconstruction, and conserves the tracer
        @test unbounded.cmin < -0.1 < -1e-3 < bounded.cmin
        @test unbounded.cmax > 1.1 > 1.05 > bounded.cmax
        @test abs(bounded.drift) < 1e-10

        # Tracers share the projection workspace, and each tracer evolves as if it were alone
        both  = advected_tracers(WENO(order=7, boundary_scheme=GhostCells()), (:c, :d)).model
        alone = advected_tracers(WENO(order=7, boundary_scheme=GhostCells()), (:d,)).model
        @test both.advection.c.buffer_scheme.projection === both.advection.d.buffer_scheme.projection
        @test Array(interior(both.tracers.c)) ≈ Array(interior(bounded.model.tracers.c))
        @test Array(interior(both.tracers.d)) ≈ Array(interior(alone.tracers.d))

        # The nonhydrostatic model projects its tracers, and advects background tracers without the projection
        model = NonhydrostaticModel(ibg; tracers = :c, advection = WENO(order=7, boundary_scheme=GhostCells()),
                                    background_fields = (; c = BackgroundField((x, z, t) -> z)))
        set!(model, u = (x, z) -> 0.1, c = initial_conditions.c)
        for _ in 1:10
            time_step!(model, 0.01)
        end
        @test all(isfinite, Array(interior(model.tracers.c)))
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
end
