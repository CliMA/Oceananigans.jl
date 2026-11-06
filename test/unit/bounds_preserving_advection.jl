include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: div_Uc, materialize_advection, update_advection!, bounds_preserving_limiter, antidiffusive_fraction
using Oceananigans.Operators: Vᶜᶜᶜ
using Random

function periodic_advection_setup(order, bounds, N; maximum_courant_number=5//18)
    grid = RectilinearGrid(CPU(), size=(N, N, N), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Periodic), halo=(6, 6, 6))

    bounds = bounds isa NTuple{2} ? BoundsPreservation(bounds...; maximum_courant_number) : bounds
    scheme = materialize_advection(WENO(; order, bounds), grid)

    return grid, scheme, CenterField(grid), (u = XFaceField(grid), v = YFaceField(grid), w = ZFaceField(grid))
end

function advect_forward_euler!(grid, scheme, c, U, N, steps, courant_number)
    ua, va, wa = 1.0, 0.7, -0.3
    set!(U.u, ua)
    set!(U.v, va)
    set!(U.w, wa)
    for field in (c, U.u, U.v, U.w)
        fill_halo_regions!(field)
    end

    Δt = courant_number / N / (abs(ua) + abs(va) + abs(wa))
    tendency = CenterField(grid)
    cᵐⁱⁿ, cᵐᵃˣ = minimum(interior(c)), maximum(interior(c))

    for step in 1:steps
        update_advection!(scheme, (; grid), c)
        for k in 1:N, j in 1:N, i in 1:N
            @inbounds tendency[i, j, k] = - div_Uc(i, j, k, grid, scheme, U, c)
        end
        c .= c .+ Δt .* tendency
        fill_halo_regions!(c)
        cᵐⁱⁿ = min(cᵐⁱⁿ, minimum(interior(c)))
        cᵐᵃˣ = max(cᵐᵃˣ, maximum(interior(c)))
    end

    return cᵐⁱⁿ, cᵐᵃˣ
end

gaussian_source(x, t) = exp(-(x - 1//4)^2 / 2 / (1//20)^2)
smooth_pulse(x, t) = ifelse(t < 1//8, 8 * exp(-(x - 1//4)^2 / 2 / (1//10)^2), zero(x))
sharp_pulse(x, t) = ifelse((t < 1//16) & (abs(x - 1//2) < 1//8), 16 * one(x), zero(x))

@testset "Bounds-preserving WENO advection" begin
    N = 8

    # These testsets evaluate the `div_Uc` stencil on the host, so they run only in the CPU job
    if child_arch isa CPU
        @testset "Conservation" begin
            for order in (5, 7, 9)
                grid, scheme, c, U = periodic_advection_setup(order, (0, 1), N)

                Random.seed!(1234)
                set!(c, (x, y, z) -> rand())
                set!(U.u, 1)
                set!(U.v, 0.7)
                set!(U.w, -0.3)
                for field in (c, U.u, U.v, U.w)
                    fill_halo_regions!(field)
                end

                update_advection!(scheme, (; grid), c)

                @test abs(sum(div_Uc(i, j, k, grid, scheme, U, c) * Vᶜᶜᶜ(i, j, k, grid)
                              for i in 1:N, j in 1:N, k in 1:N)) < 1e-12
            end
        end

        @testset "Three-dimensional bounds" begin
            sharp_cube(x, y, z) = (0.25 < x < 0.6) && (0.25 < y < 0.6) && (0.25 < z < 0.6) ? 1.0 : 0.0

            for order in (5, 7, 9)
                grid, scheme, c, U = periodic_advection_setup(order, (0, 1), N)
                set!(c, sharp_cube)
                tracer_content = sum(interior(c))

                cᵐⁱⁿ, cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 100, 5//18)

                @test cᵐⁱⁿ ≥ -1e-12
                @test cᵐᵃˣ ≤ 1 + 1e-12
                @test sum(interior(c)) ≈ tracer_content
            end

            # Without the limiter the same setup overshoots, so the bounds above are not vacuous.
            grid, scheme, c, U = periodic_advection_setup(5, nothing, N)
            set!(c, sharp_cube)
            cᵐⁱⁿ, cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 100, 5//18)

            @test cᵐⁱⁿ < -1e-6
        end

        @testset "maximum_courant_number" begin
            sharp_cube(x, y, z) = (0.25 < x < 0.6) && (0.25 < y < 0.6) && (0.25 < z < 0.6) ? 1.0 : 0.0
            courant_number = 9//20

            grid, scheme, c, U = periodic_advection_setup(5, (0, 1), N; maximum_courant_number=courant_number)
            set!(c, sharp_cube)
            cᵐⁱⁿ, cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 100, courant_number)

            @test cᵐⁱⁿ ≥ -1e-12
            @test cᵐᵃˣ ≤ 1 + 1e-12

            grid, scheme, c, U = periodic_advection_setup(5, (0, 1), N)
            set!(c, sharp_cube)
            default_cᵐⁱⁿ, default_cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 100, courant_number)

            @test default_cᵐⁱⁿ < -1e-12 || default_cᵐᵃˣ > 1 + 1e-12
        end
    end

    if child_arch isa CPU
        @testset "Extrema preservation conserves" begin
            for order in (5, 7, 9)
                grid, scheme, c, U = periodic_advection_setup(order, ExtremaPreservation(), N)

                Random.seed!(1234)
                set!(c, (x, y, z) -> rand())
                set!(U.u, 1)
                set!(U.v, 0.7)
                set!(U.w, -0.3)
                for field in (c, U.u, U.v, U.w)
                    fill_halo_regions!(field)
                end

                update_advection!(scheme, (; grid, timestepper = nothing, clock = Clock(time = 0, last_Δt = 1e-3)), c)

                @test abs(sum(div_Uc(i, j, k, grid, scheme, U, c) * Vᶜᶜᶜ(i, j, k, grid)
                              for i in 1:N, j in 1:N, k in 1:N)) < 1e-12
            end
        end

        @testset "FluxFormAdvection limits every direction" begin
            for order in (5, 7)
                grid, scheme, c, U = periodic_advection_setup(order, ExtremaPreservation(), N)
                flux_form = materialize_advection(FluxFormAdvection(WENO(; order), WENO(; order), WENO(; order)), grid)
                unlimited = materialize_advection(WENO(; order, bounds = nothing), grid)

                Random.seed!(1234)
                set!(c, (x, y, z) -> rand())
                set!(U.u, 1)
                set!(U.v, 0.7)
                set!(U.w, -0.3)
                for field in (c, U.u, U.v, U.w)
                    fill_halo_regions!(field)
                end

                model = (; grid, timestepper = nothing, clock = Clock(time = 0, last_Δt = 0.02))
                update_advection!(scheme, model, c)
                update_advection!(flux_form, model, c)

                tendency(scheme) = [div_Uc(i, j, k, grid, scheme, U, c) for i in 1:N, j in 1:N, k in 1:N]
                @test tendency(flux_form) ≈ tendency(scheme)
                @test !(tendency(unlimited) ≈ tendency(scheme))
            end
        end
    end

    @testset "Limiter is finite when the undershoot equals the regularizer" begin
        grid = RectilinearGrid(CPU(), Float32, size=16, x=(0, 1), topology=(Periodic, Flat, Flat))
        scheme = materialize_advection(WENO(Float32; order=5, bounds=(0, 1)), grid)
        c = CenterField(grid)
        c[9, 1, 1] = 4.615384f-20 # cell 8 rests on the lower bound and undershoots by ε₂ = 1e-20
        @test isfinite(bounds_preserving_limiter(8, 1, 1, grid, scheme, c))
    end

    @testset "Halo covers the limiter's extended range" begin
        @test Oceananigans.Grids.required_halo_size_x(WENO(order=5, bounds=(0, 1))) == 4
        @test Oceananigans.Grids.required_halo_size_z(WENO(order=5, bounds=(0, 1))) == 3

        grid = RectilinearGrid(CPU(), size=(4, 4, 4), extent=(1, 1, 1), halo=(3, 3, 3))
        @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; tracers=:c, tracer_advection=WENO(order=5, bounds=(0, 1)))
        @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; tracers=:c, tracer_advection=(; c=WENO(order=5, bounds=(0, 1))))
        @test_throws ArgumentError NonhydrostaticModel(grid; advection=WENO(order=5, bounds=(0, 1)), tracers=:c)
    end

    @testset "Extrema leave out immersed cells" begin
        for arch in archs
            underlying_grid = RectilinearGrid(arch, size=(16, 16, 16), x=(0, 1), y=(0, 1), z=(-1, 0),
                                              topology=(Periodic, Periodic, Bounded), halo=(6, 6, 6))
            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> x < 0.5 ? -0.5 : -1))

            c = CenterField(grid)
            set!(c, (x, y, z) -> x > 0.5 ? 20 : 10)
            r = on_architecture(arch, fill((Inf, -Inf), 1))
            extrema!(r, c)
            @test @allowscalar(r[1]) == (minimum(c), maximum(c)) == (10.0, 20.0)

            model = HydrostaticFreeSurfaceModel(grid; tracers=:c, buoyancy=nothing, tracer_advection=WENO(order=5))
            set!(model, u=(x, y, z) -> 0.1, c=(x, y, z) -> x > 0.5 ? 20 : 10)
            time_step!(model, 1e-3)

            m, M = @allowscalar model.advection.c.bounds.extrema[1]
            @test m ≥ 10 - 1e-3
            @test M ≤ 20 + 1e-3
        end
    end

    @testset "Extrema stay at the first range" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(N, N, N), x=(0, 1), y=(0, 1), z=(0, 1),
                                   topology=(Periodic, Periodic, Periodic), halo=(6, 6, 6))
            scheme = materialize_advection(WENO(order=5), grid)
            c = CenterField(grid)

            function extrema_after_update(value, last_Δt)
                set!(c, value)
                fill_halo_regions!(c)
                update_advection!(scheme, (; grid, timestepper = nothing, clock = Clock(time = 0; last_Δt)), c)
                return first(Array(scheme.bounds.extrema))
            end

            @test extrema_after_update(1, 1e-3) == (1, 1)
            @test extrema_after_update(2, 1e-3) == (1, 1)
            @test extrema_after_update(3//2, Inf) == (Inf, -Inf)
            @test extrema_after_update(3//2, 1e-3) == (3//2, 3//2)
        end
    end

    @testset "Extrema span the regions of a multi-region grid" begin
        for arch in archs
            underlying_grid = RectilinearGrid(arch, size=(2N, N, N), x=(0, 1), y=(0, 1), z=(0, 1),
                                              topology=(Periodic, Periodic, Bounded), halo=(6, 6, 6))
            multi_region_log = (:warn, "MultiRegion functionalities are experimental: help the development by reporting bugs or non-implemented features!")
            grid = @test_logs multi_region_log MultiRegionGrid(underlying_grid, partition=XPartition(2))
            model = HydrostaticFreeSurfaceModel(grid; tracers=:c, buoyancy=nothing, free_surface=ExplicitFreeSurface(),
                                                tracer_advection=WENO(order=5))
            model.clock.last_Δt = 1e-3

            function extrema_after_update(left_value, right_value)
                set!(model, c=(x, y, z) -> x < 1/2 ? left_value : right_value)
                return first(Array(model.advection.c.bounds.extrema))
            end

            @test extrema_after_update(1, 2) == (1, 2)
            @test extrema_after_update(3, 3//2) == (1, 2)
        end
    end

    @testset "The extrema widen by the cell, what flows into it and the height of a smooth extremum" begin
        κ, α, E = 0.1, 0.7, (0.0, 1.0)
        raising_fraction(cᶜ, inflow, cᶠ; ε = 0.0) = antidiffusive_fraction(cᶜ, κ, 1.0, (inflow, cᶠ), (0.0, cᶜ), false, α, E, (0.0, ε))
        lowering_fraction(cᶜ, inflow, cᶠ; ε = 0.0) = antidiffusive_fraction(cᶜ, κ, 1.0, (inflow, cᶠ), (0.0, cᶜ), true, α, E, (ε, 0.0))

        # within the extrema the distance to them sets the fraction, 0.05 / 0.1
        @test raising_fraction(0.95, 0.0, 1.0) ≈ 0.5
        @test lowering_fraction(0.05, 0.0, 0.0) ≈ 0.5

        # beyond the extrema a neighbor widens them only through a flow that brings it in
        @test raising_fraction(2.0, 1.0, 3.0) == 1
        @test raising_fraction(2.0, 0.0, 3.0) == 0
        @test raising_fraction(2.0, -1.0, 3.0) == 0
        @test lowering_fraction(-2.0, 1.0, -3.0) == 1
        @test lowering_fraction(-2.0, 0.0, -3.0) == 0

        # the height of a smooth extremum widens them by 0.05 of the 0.1 the antidiffusive flux asks for
        @test raising_fraction(2.0, 0.0, 3.0; ε = 0.05) ≈ 0.5
        @test lowering_fraction(-2.0, 0.0, -3.0; ε = 0.05) ≈ 0.5
    end

    @testset "A source from zero keeps the high-order reconstruction" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=4N, x=(0, 1), topology=(Periodic, Flat, Flat), halo=6)
            velocities = PrescribedVelocityFields(u=1)
            source = Forcing(gaussian_source)
            Δt = 1 / 4 / (4N)

            function advected_variance(tracer_advection)
                model = HydrostaticFreeSurfaceModel(grid; velocities, tracers=:c, buoyancy=nothing, tracer_advection,
                                                    forcing=(; c=source))
                for _ in 1:40N
                    time_step!(model, Δt)
                end
                c = Array(interior(model.tracers.c))
                return sum(abs2, c .- sum(c) / length(c)), minimum(c)
            end

            weno_variance, weno_minimum = advected_variance(WENO(order=5))
            upwind_variance, _ = advected_variance(UpwindBiased(order=1))
            @test weno_variance > 1.1 * upwind_variance
            @test weno_minimum > -1e-3 * sqrt(weno_variance)
        end
    end

    @testset "An advected extremum beyond the extrema keeps its height" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=4N, x=(0, 1), topology=(Periodic, Flat, Flat), halo=6)
            velocities = PrescribedVelocityFields(u=1)
            Δt = 1 / 4 / (4N)

            function advected_maximum(pulse, tracer_advection)
                model = HydrostaticFreeSurfaceModel(grid; velocities, tracers=:c, buoyancy=nothing, tracer_advection,
                                                    forcing=(; c=Forcing(pulse)))
                for _ in 1:40N
                    time_step!(model, Δt)
                end
                return maximum(Array(interior(model.tracers.c)))
            end

            # a smooth extremum rises and falls as it crosses the cells, and a step does not rise
            @test advected_maximum(smooth_pulse, WENO(order=5)) > 0.99 * advected_maximum(smooth_pulse, WENO(order=5, bounds=nothing))
            @test advected_maximum(sharp_pulse, WENO(order=5)) < 1 + 1e-3
        end
    end

    @testset "A tracer enters through an Open side with its boundary value" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=8N, x=(0, 1), topology=(Open, Flat, Flat), halo=4)
            velocities = PrescribedVelocityFields(u=1)
            boundary_conditions = (; c=FieldBoundaryConditions(west=ValueBoundaryCondition(1)))
            Δt = 1 / 5 / (8N)

            function error_behind_front(tracer_advection)
                model = HydrostaticFreeSurfaceModel(grid; velocities, tracers=:c, buoyancy=nothing, tracer_advection, boundary_conditions)
                for _ in 1:25N
                    time_step!(model, Δt)
                end
                c = Array(interior(model.tracers.c, :, 1, 1))
                x = Array(xnodes(grid, Center()))
                return maximum(abs, c[x .< 25N * Δt - 0.15] .- 1)
            end

            @test error_behind_front(WENO(order=7)) < 1e-4
            @test error_behind_front(WENO(order=7, bounds=nothing)) < 1e-4
        end
    end

    @testset "Model integration" begin
        for arch in archs
            underlying_grid = RectilinearGrid(arch, size=(N, N, N), x=(0, 1), y=(0, 1), z=(0, 1),
                                              topology=(Periodic, Periodic, Bounded), halo=(6, 6, 6))

            for model_type in (:nonhydrostatic, :hydrostatic), immersed in (false, true), bounds in ((0, 1), ExtremaPreservation())
                grid = immersed ?
                    ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> x < 0.5 ? -0.5 : -1)) :
                    underlying_grid

                advection = WENO(order=5; bounds)

                model = if model_type === :nonhydrostatic
                    NonhydrostaticModel(grid; advection, tracers=:c)
                else
                    HydrostaticFreeSurfaceModel(grid; tracers=:c,
                                                momentum_advection = WENO(order=5),
                                                tracer_advection = advection)
                end

                set!(model, u=(x, y, z) -> 0.1 * randn(), c=(x, y, z) -> x > 0.5 ? 1.0 : 0.0)
                time_step!(model, 1e-3)

                @test all(isfinite, interior(model.tracers.c))
            end
        end
    end
end
