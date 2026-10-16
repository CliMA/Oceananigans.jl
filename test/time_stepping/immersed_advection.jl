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

# Schemes wider than the lowest order are reduced in the z-direction of grids with a single cell
reduced_z_advection_logs(adv, buffer) =
    buffer == 1      ? () :
    adv == Centered  ? ((:info, "Using the advection scheme Centered(order=2) in the z-direction because size(grid, 3) = 1"),) :
                       ((:info, "Using the advection scheme UpwindBiased(order=1) in the z-direction because size(grid, 3) = 1"),)

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
        grid = RectilinearGrid(arch, size=(20, 20), extent=(20, 20), halo = (6, 6), topology=(Bounded, Bounded, Flat))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y) -> (x < 5 || y < 5)))

        c = CenterField(ibg)
        set!(c, 1)
        mask_immersed_field!(c)
        fill_halo_regions!(c)

        for adv in linear_advection_schemes, buffer in [1, 2, 3, 4, 5]
            scheme = adv(order = advective_order(buffer, adv))

            @testset "$(summary(scheme)) [$(summary(arch))]" begin
                run_tracer_interpolation_test(c, ibg, scheme)
            end
        end

        for buffer in [2, 3, 4, 5], bounds in (nothing, (0, 1))
            scheme = WENO(; order = advective_order(buffer, WENO), bounds)

            @testset "$(summary(scheme)) [$(summary(arch))]" begin
                run_tracer_interpolation_test(c, ibg, scheme)
            end
        end
    end

    @testset "Immersed tracer conservation" begin
        grid = RectilinearGrid(arch, size=(10, 8, 1), extent=(10, 8, 1), halo = (6, 6, 6), topology=(Bounded, Periodic, Bounded))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> ifelse(x < 2, 0, -1)))

        for adv in advection_schemes, buffer in [1, 2, 3, 4, 5]
            scheme = adv(order = advective_order(buffer, adv))

            @testset "$(summary(scheme)) on $(nameof(typeof(g))) [$(summary(arch))]" for g in [grid, ibg]
                @test_logs reduced_z_advection_logs(adv, buffer)... run_tracer_conservation_test(g, scheme)
            end
        end

        for adv in advection_schemes, buffer in [1, 2, 3, 4, 5]
            directional_scheme = adv(order = advective_order(buffer, adv))
            scheme = FluxFormAdvection(directional_scheme, directional_scheme, directional_scheme)

            @testset "$(summary(scheme)) on $(nameof(typeof(g))) [$(summary(arch))]" for g in [grid, ibg]
                @test_logs reduced_z_advection_logs(adv, buffer)... run_tracer_conservation_test(g, scheme)
            end
        end
    end

    @testset "GhostCells boundary reconstruction" begin
        scheme = WENO(order=7, weight_computation=NormalDivision, boundary_scheme=GhostCells()).buffer_scheme

        function reconstruct(scheme, bias, values, activity)
            ψ̂, ghosted_face = ghost_cell_reconstruction(scheme.scheme, values, activity, bias, nothing)
            return ψ̂
        end

        # Averages over the unit cells [n - 1, n] of a polynomial with primitive P; the reconstructed face is at x = 4
        averages(P) = ntuple(n -> P(n) - P(n - 1), 8)

        active   = ntuple(_ -> true, 8)
        wall     = (false, false, true, true, true, true, true, true)
        two_left = (false, false, false, true, true, false, false, false)

        linear_primitive(x) = x^2 / 2 + 2x # p(x) = x + 2

        for bias in (LeftBias, RightBias)
            # the values in inactive cells never enter the reconstruction
            @test reconstruct(scheme, bias, (1e3, 1e3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), wall) == 3.0
            @test reconstruct(scheme, bias, Tuple(1.0:8.0), active) == 4.5

            # the ghosts extrapolate linearly, so a uniform stratification next to a wall is reconstructed exactly
            @test reconstruct(scheme, bias, averages(linear_primitive), wall) ≈ 4 + 2
            @test reconstruct(scheme, bias, averages(linear_primitive), two_left) ≈ 4 + 2
        end


        # the ghosts lie on the line through the active cells in index coordinates, those of the WENO reconstruction
        affine = Tuple(2.5 .- 0.7 .* (1:8))
        lowest_inactive = (false, true, true, true, true, true, true, true)
        @test reconstruct(scheme, LeftBias, (1e3, affine[2:8]...), lowest_inactive) ≈ reconstruct(scheme, LeftBias, affine, active)

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

    @testset "GhostCells tracer advection" begin
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

        function advected_tracer(scheme)
            model = HydrostaticFreeSurfaceModel(ibg; velocities = PrescribedVelocityFields(; u, w), tracers = :c,
                                                tracer_advection = scheme, timestepper = :SplitRungeKutta3)
            set!(model, c = (x, z) -> ifelse(-0.6 < z < -0.4, 1.0, 0.0))
            c = model.tracers.c
            total₀ = sum(interior(c))
            cmin, cmax = 0.0, 1.0
            for _ in 1:100
                time_step!(model, 0.01)
                cmin = min(cmin, minimum(interior(c)))
                cmax = max(cmax, maximum(interior(c)))
            end
            return (; cmin, cmax, drift = sum(interior(c)) - total₀)
        end

        ghost_cells = advected_tracer(WENO(order=7, boundary_scheme=GhostCells()))
        chain = advected_tracer(WENO(order=7))

        # Next to the walls the clamped face values remove most of the extrema of the default buffer chain
        @test chain.cmin < -0.2 < -0.02 < ghost_cells.cmin
        @test chain.cmax > 1.4 > 1.05 > ghost_cells.cmax
        @test abs(ghost_cells.drift) < 1e-10
    end

    @testset "GhostCells on a z-star grid" begin
        Nx, Nz = 64, 32
        grid = RectilinearGrid(arch, size=(Nx, Nz), x=(-1e6, 1e6), z=MutableVerticalDiscretization((-2000, 0)), halo=(6, 6),
                               topology=(Periodic, Flat, Bounded))
        ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(x -> -2000 + 750 * exp(-x^2 / 8e8)))
        model = HydrostaticFreeSurfaceModel(ibg; timestepper = :SplitRungeKutta3, buoyancy = BuoyancyTracer(), tracers = :b,
                                            tracer_advection = WENO(order=7, boundary_scheme=GhostCells()),
                                            momentum_advection = WENO(order=5, boundary_scheme=GhostCells()))
        set!(model, u = (x, z) -> 0.3 * exp(-x^2 / 1e10), b = (x, z) -> 1e-4 * z)
        for _ in 1:20
            time_step!(model, 600)
        end

        b = Array(interior(model.tracers.b))
        @test @allowscalar maximum(abs, model.free_surface.displacement) > 0.1
        @test all(isfinite, b)
        @test -0.2 - 1e-10 ≤ minimum(b) && maximum(b) ≤ 1e-10
    end

    @testset "Immersed momentum reconstruction" begin
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

            @testset "$(summary(scheme)) [$(summary(arch))]" begin
                run_momentum_interpolation_test(u, v, ibg, scheme)
            end
        end
    end
end
