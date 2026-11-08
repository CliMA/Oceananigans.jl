include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: div_Uc, materialize_advection, update_advection!, update_flux_limiter!, bounds_preserving_limiter
using Oceananigans.Operators: Vᶜᶜᶜ
using Random

function periodic_advection_setup(order, bounds, N; maximum_cfl=5//18, flux_limiter=false)
    grid = RectilinearGrid(CPU(), size=(N, N, N), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Periodic), halo=(6, 6, 6))

    scheme = materialize_advection(WENO(; order, bounds, maximum_cfl, flux_limiter), grid)

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
        update_flux_limiter!(scheme, grid, c, U, Δt) # a no-op without the flux limiter
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

        @testset "maximum_cfl" begin
            sharp_cube(x, y, z) = (0.25 < x < 0.6) && (0.25 < y < 0.6) && (0.25 < z < 0.6) ? 1.0 : 0.0
            courant_number = 9//20

            grid, scheme, c, U = periodic_advection_setup(5, (0, 1), N; maximum_cfl=courant_number)
            set!(c, sharp_cube)
            cᵐⁱⁿ, cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 100, courant_number)

            @test cᵐⁱⁿ ≥ -1e-12
            @test cᵐᵃˣ ≤ 1 + 1e-12

            grid, scheme, c, U = periodic_advection_setup(5, (0, 1), N)
            set!(c, sharp_cube)
            default_cᵐⁱⁿ, default_cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 100, courant_number)

            @test default_cᵐⁱⁿ < -1e-12 || default_cᵐᵃˣ > 1 + 1e-12
        end

        @testset "Flux limiter" begin
            sharp_cube(x, y, z) = (0.25 < x < 0.6) && (0.25 < y < 0.6) && (0.25 < z < 0.6) ? 1.0 : 0.0
            sharp_hole(x, y, z) = 1 - sharp_cube(x, y, z)
            courant_number = 0.5 # above ω̂₁ = 5//18, where the rescaling alone no longer guarantees the bounds

            # The rescaling alone undershoots below the cube and, symmetrically, overshoots above the hole
            grid, scheme, c, U = periodic_advection_setup(5, (0, 1), N)
            set!(c, sharp_cube)
            cᵐⁱⁿ, cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 50, courant_number)
            @test cᵐⁱⁿ < -1e-6

            grid, scheme, c, U = periodic_advection_setup(5, (0, 1), N)
            set!(c, sharp_hole)
            cᵐⁱⁿ, cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 50, courant_number)
            @test cᵐᵃˣ > 1 + 1e-6

            for initial_condition in (sharp_cube, sharp_hole)
                grid, scheme, c, U = periodic_advection_setup(5, (0, 1), N; flux_limiter=true)
                set!(c, initial_condition)
                tracer_content = sum(interior(c))
                cᵐⁱⁿ, cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 50, courant_number)
                @test cᵐⁱⁿ ≥ -1e-12
                @test cᵐᵃˣ ≤ 1 + 1e-12
                @test sum(interior(c)) ≈ tracer_content
            end

            # A nonzero lower bound: the limiter acts on the fluxes of c - cᵐⁱⁿ
            grid, scheme, c, U = periodic_advection_setup(5, (2, 3), N; flux_limiter=true)
            set!(c, (x, y, z) -> 2 + sharp_cube(x, y, z))
            tracer_content = sum(interior(c))
            cᵐⁱⁿ, cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 50, courant_number)
            @test cᵐⁱⁿ ≥ 2 - 1e-12
            @test cᵐᵃˣ ≤ 3 + 1e-12
            @test sum(interior(c)) ≈ tracer_content

            grid, scheme, c, U = periodic_advection_setup(5, (2, 3), N)
            set!(c, (x, y, z) -> 2 + sharp_cube(x, y, z))
            cᵐⁱⁿ, cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 50, courant_number)
            @test cᵐⁱⁿ < 2 - 1e-6
        end
    end

    @testset "Flux limiter options" begin
        @test_throws ArgumentError WENO(order=5, flux_limiter=true) # no bounds
        @test_throws ArgumentError WENO(order=5, bounds=(0, 1), flux_limiter=:limited)
        @test WENO(order=5, bounds=(-1, 1), flux_limiter=true).bounds.minimum_value == -1
    end

    @testset "Grouped bounds constructor" begin
        grouped = WENO(order=5, bounds=BoundsPreservation(0, 1; maximum_cfl=1//4, flux_limiter=FluxLimiter()))
        flat = WENO(order=5, bounds=(0, 1), maximum_cfl=1//4, flux_limiter=FluxLimiter())
        @test grouped.bounds == flat.bounds
        @test grouped.bounds.maximum_cfl === 0.25
        @test grouped.bounds.flux_limiter isa FluxLimiter
        @test grouped.buffer_scheme.bounds == flat.buffer_scheme.bounds
        @test WENO(order=5, bounds=BoundsPreservation(0, 1; flux_limiter=true)).bounds == WENO(order=5, bounds=(0, 1), flux_limiter=true).bounds
        @test WENO(Float32; order=5, bounds=BoundsPreservation(0, 1)).bounds.maximum_value isa Float32

        @test_throws ArgumentError WENO(order=5, bounds=BoundsPreservation(0, 1), flux_limiter=true)
        @test_throws ArgumentError WENO(order=5, bounds=BoundsPreservation(0, 1), maximum_cfl=1//4)
        @test_throws ArgumentError BoundsPreservation(1, 0)
        @test_throws ArgumentError BoundsPreservation(0, 1; maximum_cfl=1//2)

        deprecated = @test_logs (:warn, r"maximum_courant_number") match_mode=:any WENO(order=5, bounds=(0, 1), maximum_courant_number=1//4)
        @test deprecated.bounds.maximum_cfl === 0.25
    end

    # In a uniform column falling at a vertical Courant number ν > 1 under forward Euler, every cell with
    # outflow is limited to the fraction r = 1/ν its content affords, and the update stays non-negative
    # and conserves the tracer.
    @testset "Fractions of a uniform column at vertical Courant number 1.5" begin
        Nz = 16
        grid = RectilinearGrid(CPU(), size=Nz, z=(0, Nz), topology=(Flat, Flat, Bounded), halo=4)
        ν = 1.5

        function fractions_and_tendency(flux_limiter)
            scheme = materialize_advection(WENO(; order=5, bounds=(0, 4), flux_limiter), grid)
            c = CenterField(grid)
            set!(c, 1)
            fill_halo_regions!(c)
            U = (u = XFaceField(grid), v = YFaceField(grid), w = ZFaceField(grid))
            set!(U.w, -ν)
            interior(U.w)[1, 1, 1] = 0    # impenetrable bottom
            interior(U.w)[1, 1, Nz+1] = 0 # impenetrable top
            fill_halo_regions!(U.w)
            update_advection!(scheme, (; grid), c)
            update_flux_limiter!(scheme, grid, c, U, 1)
            r = copy(interior(scheme.bounds.flux_limiter.outgoing_fraction)[1, 1, :])
            tendency = [-div_Uc(1, 1, k, grid, scheme, U, c) for k in 1:Nz]
            return r, tendency
        end

        r, tendency = fractions_and_tendency(true)
        @test all(r[2:Nz] .≈ 1 / ν) # every cell with outflow is limited by its content alone
        @test r[1] == 1              # the bottom cell has no outflow
        @test minimum(1 .+ tendency) ≥ -1e-12
        @test abs(sum(tendency)) < 1e-12
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

            for model_type in (:nonhydrostatic, :hydrostatic), immersed in (false, true), flux_limiter in (false, true)
                grid = immersed ?
                    ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> x < 0.5 ? -0.5 : -1)) :
                    underlying_grid

                advection = WENO(order=5, bounds=(0, 1), flux_limiter=flux_limiter)

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
