include("dependencies_for_runtests.jl")

using Oceananigans.Advection: div_Uc, materialize_advection, update_advection!, compute_bounds_preserving_limiter!
using Oceananigans.BoundaryConditions: fill_halo_regions!
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

    @testset "Limiter at the lower bound with an undershoot of exactly ε₂" begin
        # A tracer that has decayed to zero below a tail of ε₂-multiples: the right-biased
        # reconstruction at the face below the zero cell is exactly -ε₂ (0.3 × (-1/6) × 2e-19),
        # which used to give θ = |0 / (m - cᵢ + ε₂)| = 0/0. Taken from a Float32 P3 rain-number
        # field at cloud top.
        FT = Float32
        grid = RectilinearGrid(CPU(), FT; size=(8, 8, 12), x=(0, 8), y=(0, 8), z=(0, 12), halo=(5, 5, 5),
                               topology=(Periodic, Periodic, Bounded))
        column = FT[1.45, 9.531033f-13, 2.0000002f-19, 4.6913728f-29, 0, 0, 0, 0, 0, 0, 0, 0]
        for bounds in ((0.0, 1.0), (0.0, Inf))
            scheme = materialize_advection(WENO(FT; order=5, bounds), grid)
            c = CenterField(grid)
            set!(c, (x, y, z) -> column[clamp(Int(floor(z)) + 1, 1, 12)])
            fill_halo_regions!(c)
            compute_bounds_preserving_limiter!(scheme, grid, c)
            θ = Array(interior(scheme.bounds.limiter))
            @test all(isfinite, θ)
            @test all(θ[:, :, 5:end] .== 0)   # zero cells at the lower bound admit no excursion
        end
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

    @testset "Model integration" begin
        for model_type in (:nonhydrostatic, :hydrostatic), immersed in (false, true)
            underlying_grid = RectilinearGrid(CPU(), size=(N, N, N), x=(0, 1), y=(0, 1), z=(0, 1),
                                              topology=(Periodic, Periodic, Bounded), halo=(6, 6, 6))

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
