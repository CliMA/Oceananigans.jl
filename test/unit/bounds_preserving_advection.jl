include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: div_Uc, materialize_advection, update_advection!, bounds_preserving_limiter, update_bounds_preserving_limiter!
using Oceananigans.TimeSteppers: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.Operators: Vᶜᶜᶜ
using Random

function periodic_advection_setup(order, bounds, N; maximum_courant_number=5//18)
    grid = RectilinearGrid(CPU(), size=(N, N, N), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Periodic), halo=(6, 6, 6))

    scheme = materialize_advection(WENO(; order, bounds, maximum_courant_number), grid)

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

    @testset "Model integration" begin
        for arch in archs
            underlying_grid = RectilinearGrid(arch, size=(N, N, N), x=(0, 1), y=(0, 1), z=(0, 1),
                                              topology=(Periodic, Periodic, Bounded), halo=(6, 6, 6))

            for model_type in (:nonhydrostatic, :hydrostatic), immersed in (false, true)
                grid = immersed ?
                    ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> x < 0.5 ? -0.5 : -1)) :
                    underlying_grid

                advection = WENO(order=5, bounds=(0, 1))

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

# With adaptive-implicit vertical advection, the bounded divergence must advect only the explicit,
# CFL-limited part of `w`: the implicit solve adds the rest. Far from its bounds (θ = 1) the bounded
# scheme must then reproduce the unbounded one, including where the vertical Courant number exceeds `cfl`.
@testset "Bounded WENO with adaptive-implicit vertical advection" begin
    N = 16
    grid = RectilinearGrid(CPU(), size=(8, 8, N), x=(0, 1), y=(0, 1), z=(-1, 0),
                           topology=(Periodic, Periodic, Bounded), halo=(6, 6, 6))

    Δt = 0.05
    tds = map(1:2) do _
        td = AdaptiveVerticallyImplicitDiscretization(cfl=0.5)
        td.Δt[] = Δt
        td
    end

    bounded   = materialize_advection(WENO(order=5, bounds=(0, 100), time_discretization=tds[1]), grid)
    unbounded = materialize_advection(WENO(order=5, time_discretization=tds[2]), grid)

    c = CenterField(grid)
    set!(c, (x, y, z) -> 34 + 0.5 * sin(2π * x) + 0.3 * cos(3π * z))
    fill_halo_regions!(c)

    U = (u = XFaceField(grid), v = YFaceField(grid), w = ZFaceField(grid))
    set!(U.u, (x, y, z) -> 0.2 * cos(2π * y))
    set!(U.w, (x, y, z) -> -1.5 * sin(π * z)) # |w| Δt / Δz reaches 1.2, above cfl = 0.5
    foreach(fill_halo_regions!, U)

    update_bounds_preserving_limiter!(bounded, grid, c)

    max_difference = 0.0
    for k in 1:N, j in 1:8, i in 1:8
        db = div_Uc(i, j, k, grid, bounded,   U, c)
        du = div_Uc(i, j, k, grid, unbounded, U, c)
        max_difference = max(max_difference, abs(db - du))
    end

    @test max_difference < 1e-10
end
