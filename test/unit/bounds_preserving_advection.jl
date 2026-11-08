include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: div_Uc, materialize_advection, update_advection!, update_flux_limiter!, bounds_preserving_limiter
using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedVelocityFields
using Oceananigans.Operators: Vᶜᶜᶜ
using Random

function periodic_advection_setup(order, bounds, N; maximum_courant_number=5//18, flux_limiter=false)
    grid = RectilinearGrid(CPU(), size=(N, N, N), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Periodic), halo=(6, 6, 6))

    scheme = materialize_advection(WENO(; order, bounds, maximum_courant_number, flux_limiter), grid)

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
        update_flux_limiter!(scheme, grid, c, c, U, Δt) # a no-op without the flux limiter
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

        @testset "Flux limiter" begin
            sharp_cube(x, y, z) = (0.25 < x < 0.6) && (0.25 < y < 0.6) && (0.25 < z < 0.6) ? 1.0 : 0.0
            courant_number = 0.5 # above ω̂₁ = 5//18, where the rescaling alone no longer guarantees the bounds

            grid, scheme, c, U = periodic_advection_setup(5, (0, 1), N)
            set!(c, sharp_cube)
            cᵐⁱⁿ, cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 50, courant_number)
            @test cᵐⁱⁿ < -1e-6

            grid, scheme, c, U = periodic_advection_setup(5, (0, 1), N; flux_limiter=true)
            set!(c, sharp_cube)
            tracer_content = sum(interior(c))
            cᵐⁱⁿ, cᵐᵃˣ = advect_forward_euler!(grid, scheme, c, U, N, 50, courant_number)
            @test cᵐⁱⁿ ≥ -1e-12
            @test sum(interior(c)) ≈ tracer_content
        end
    end

    @testset "Flux limiter requires a lower bound of zero" begin
        @test_throws ArgumentError WENO(order=5, flux_limiter=true)
        @test_throws ArgumentError WENO(order=5, bounds=(-1, 1), flux_limiter=true)
        @test_throws ArgumentError WENO(order=5, bounds=(-1, 1), flux_limiter=FluxLimiter(credit=ColumnSweep()))
        @test_throws ArgumentError WENO(order=5, bounds=(0, 1), flux_limiter=:column_sweep)
    end

    # A pulse falling at a vertical Courant number above one: the single pass credits no incoming flux,
    # so it caps the transport at one cell per update and withholds most of the pulse; the column sweep
    # credits the cell above and lets the pulse fall. Both stay non-negative and conserve the tracer.
    @testset "Column sweep at vertical Courant number 1.2" begin
        Nz = 160
        grid = RectilinearGrid(CPU(), size=Nz, z=(0, Nz), topology=(Flat, Flat, Periodic), halo=4)
        pulse(z) = exp(-((z - 110.5) / 4)^2)
        courant_number = 1.2
        steps = 60
        shift = round(Int, courant_number * steps)

        function fall_with_split_rk3(flux_limiter)
            scheme = materialize_advection(WENO(; order=5, bounds=(0, 1), flux_limiter), grid)
            c = CenterField(grid)
            set!(c, pulse)
            parent(c) ./= sum(interior(c))
            tracer_content = sum(interior(c))
            U = (u = XFaceField(grid), v = YFaceField(grid), w = ZFaceField(grid))
            set!(U.w, -courant_number)
            fill_halo_regions!(U.w)
            c⁰ = CenterField(grid)
            tendency = CenterField(grid)
            cᵐⁱⁿ = 0.0

            for step in 1:steps
                parent(c⁰) .= parent(c)
                for β in (3, 2, 1) # Wicker–Skamarock stages from c⁰, with fluxes built from the previous stage
                    Δτ = 1 / β
                    fill_halo_regions!(c)
                    update_advection!(scheme, (; grid), c)
                    update_flux_limiter!(scheme, grid, c, c⁰, U, Δτ)
                    for k in 1:Nz
                        @inbounds tendency[1, 1, k] = - div_Uc(1, 1, k, grid, scheme, U, c)
                    end
                    interior(c) .= interior(c⁰) .+ Δτ .* interior(tendency)
                    cᵐⁱⁿ = min(cᵐⁱⁿ, minimum(interior(c)))
                end
            end

            return c, cᵐⁱⁿ, sum(interior(c)) - tracer_content
        end

        exact = CenterField(grid)
        set!(exact, pulse)
        parent(exact) ./= sum(interior(exact))
        analytic = circshift(interior(exact)[1, 1, :], -shift)

        c, cᵐⁱⁿ, drift = fall_with_split_rk3(true)
        single_pass_error = sum(abs.(interior(c)[1, 1, :] .- analytic))
        @test cᵐⁱⁿ ≥ -1e-12
        @test abs(drift) < 1e-12

        c, cᵐⁱⁿ, drift = fall_with_split_rk3(FluxLimiter(credit=ColumnSweep()))
        column_sweep_error = sum(abs.(interior(c)[1, 1, :] .- analytic))
        @test cᵐⁱⁿ ≥ -1e-12
        @test abs(drift) < 1e-12

        @test single_pass_error > 1.5    # the pulse is held back: no overlap with where it should be
        @test column_sweep_error < 0.5   # the pulse falls, with the scheme's own error
    end

    # The stages of a split Runge-Kutta step update the step-start state with fluxes built from a later
    # stage, so the rescaling alone does not keep the tracer non-negative even at small Courant numbers.
    @testset "Flux limiter with SplitRungeKutta3" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(N, N, N), x=(0, 1), y=(0, 1), z=(-1, 0),
                                   topology=(Periodic, Periodic, Bounded), halo=(6, 6, 6))
            sharp_cube(x, y, z) = (0.25 < x < 0.6) && (0.25 < y < 0.6) && (-0.75 < z < -0.4) ? 1.0 : 0.0

            function advect_with_split_rk3(tracer_advection)
                velocities = PrescribedVelocityFields(u=1, v=0.7)
                model = HydrostaticFreeSurfaceModel(grid; velocities, tracers=:c, buoyancy=nothing,
                                                    tracer_advection, timestepper=:SplitRungeKutta3)
                set!(model, c=sharp_cube)
                tracer_content = sum(interior(model.tracers.c))
                Δt = 0.6 / N / 1.7 # Courant number 0.6
                cᵐⁱⁿ = zero(eltype(grid))
                for step in 1:30
                    time_step!(model, Δt)
                    cᵐⁱⁿ = min(cᵐⁱⁿ, minimum(interior(model.tracers.c)))
                end
                return cᵐⁱⁿ, sum(interior(model.tracers.c)) - tracer_content
            end

            cᵐⁱⁿ, drift = advect_with_split_rk3(WENO(order=5, bounds=(0, 1)))
            @test cᵐⁱⁿ < -1e-8

            cᵐⁱⁿ, drift = advect_with_split_rk3(WENO(order=5, bounds=(0, 1), flux_limiter=true))
            @test cᵐⁱⁿ ≥ -1e-12
            @test abs(drift) < 1e-10
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
