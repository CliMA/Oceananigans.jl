using Oceananigans
using Oceananigans.Advection: U_dot_∇u_metric, U_dot_∇v_metric, U_dot_∇w_metric
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Test

#####
##### Tests for curvature metric terms in flux-form momentum advection
#####
##### Solid body rotation u = u₀ cos(φ), v = w = 0 is a steady-state solution
##### on the sphere when Coriolis and the metric (curvature / centrifugal)
##### terms balance the meridional pressure gradient. The unit test below
##### checks the metric functions directly against analytic centrifugal values:
##### removing or zero-ing the metric terms makes those tests fail by orders
##### of magnitude. The simulation tests are coarser sanity checks that the
##### terms are wired into the tendency kernels with the right sign.
#####

@testset "Curvature metric terms" begin
    for arch in archs

        #####
        ##### Direct unit test: U_dot_∇{u,v,w}_metric on solid body rotation
        #####
        ##### For u = u₀ cos(φ), v = w = 0 the metric functions evaluate to:
        #####     U_dot_∇u_metric ≈ 0
        #####     U_dot_∇v_metric ≈ +u² tan(φ) / a       (= -G_v, the centrifugal pull
        #####                                              toward the equator subtracted
        #####                                              from the v tendency)
        #####     U_dot_∇w_metric ≈ -u² / a              (= -G_w, the upward centrifugal
        #####                                              subtracted from the w tendency)
        ##### If the curvature metric is removed from the source code, all three
        ##### functions return 0 and the v/w assertions fail by 10+ orders of magnitude.
        #####
        @testset "Analytic centrifugal at solid body rotation [$(typeof(arch))]" begin
            grid = LatitudeLongitudeGrid(arch;
                                         size = (36, 34, 4),
                                         halo = (4, 4, 4),
                                         longitude = (0, 360),
                                         latitude = (-80, 80),
                                         z = (-1000, 0))

            u = XFaceField(grid)
            v = YFaceField(grid)
            w = ZFaceField(grid)

            u₀ = 10
            set!(u, (λ, φ, z) -> u₀ * cosd(φ))
            fill_halo_regions!(u)

            U = (u, v, w)
            advection = WENO(order=5)

            a = grid.radius
            k = 2 # any interior level

            # u-metric: should vanish for pure zonal flow
            @allowscalar for j in 8:grid.Ny-7, i in 8:grid.Nx-7
                metric = U_dot_∇u_metric(i, j, k, grid, advection, U, U)
                @test abs(metric) < 1e-10 * u₀^2 / a
            end

            # v-metric: should equal +u² tan(φ) / a at the v point
            @allowscalar for j in 8:grid.Ny-7, i in 8:grid.Nx-7
                φ = φnode(i, j, k, grid, Center(), Face(), Center())
                expected = (u₀ * cosd(φ))^2 * tand(φ) / a
                actual = U_dot_∇v_metric(i, j, k, grid, advection, U, U)
                @test isapprox(actual, expected; rtol=0.02, atol=1e-12)
            end

            # w-metric: should equal -u² / a at the w point
            @allowscalar for j in 8:grid.Ny-7, i in 8:grid.Nx-7
                φ = φnode(i, j, k, grid, Center(), Center(), Face())
                expected = -(u₀ * cosd(φ))^2 / a
                actual = U_dot_∇w_metric(i, j, k, grid, advection, U, U)
                @test isapprox(actual, expected; rtol=0.02)
            end
        end

        #####
        ##### Integration test: NonhydrostaticModel solid body rotation
        ##### Uses SphericalCoriolis (full 3D, with the 2Ω cos(φ) horizontal-rotation
        ##### terms that couple u and w) — appropriate for the nonhydrostatic
        ##### momentum equations and consistent with the nonhydrostatic metric corrections.
        #####
        @testset "Solid body rotation [NonhydrostaticModel, $(typeof(arch))]" begin
            grid = LatitudeLongitudeGrid(arch;
                                         size = (36, 34, 4),
                                         halo = (4, 4, 4),
                                         longitude = (0, 360),
                                         latitude = (-80, 80),
                                         z = (-1000, 0))

            coriolis = SphericalCoriolis()
            pressure_solver = ConjugateGradientPoissonSolver(grid; reltol=1e-7)

            model = NonhydrostaticModel(grid; coriolis, pressure_solver,
                                        advection = WENO(order=5))

            u₀ = 10
            set!(model, u = (λ, φ, z) -> u₀ * cosd(φ))

            simulation = Simulation(model; Δt = 10, stop_iteration = 10)
            run!(simulation)

            v_max = maximum(abs, interior(model.velocities.v))
            w_max = maximum(abs, interior(model.velocities.w))
            u_max = maximum(abs, interior(model.velocities.u))

            @test v_max < 1   # ≪ u₀
            @test w_max < 1
            @test u_max < 2 * u₀
            @test !any(isnan, interior(model.velocities.u))
            @test !any(isnan, interior(model.velocities.v))
            @test !any(isnan, interior(model.velocities.w))
        end

        #####
        ##### Integration test: HydrostaticFreeSurfaceModel solid body rotation
        ##### with analytic balanced free surface.
        #####
        @testset "Solid body rotation [HydrostaticFreeSurfaceModel, $(typeof(arch))]" begin
            grid = LatitudeLongitudeGrid(arch;
                                         size = (36, 34, 4),
                                         halo = (4, 4, 4),
                                         longitude = (0, 360),
                                         latitude = (-80, 80),
                                         z = (-1000, 0))

            free_surface = SplitExplicitFreeSurface(grid; substeps=10)
            coriolis = HydrostaticSphericalCoriolis()

            model = HydrostaticFreeSurfaceModel(grid; coriolis, free_surface,
                                                momentum_advection = WENOVectorInvariant(order=5))

            u₀ = 1//10 # 0.1 m/s; small enough that η stays well within layer depth
            g = model.free_surface.gravitational_acceleration
            R = grid.radius
            Ω = model.coriolis.rotation_rate

            # Balanced free surface for solid body rotation:
            # geostrophic + cyclostrophic balance gives ∂η/∂φ = -a(fu + u² tan(φ)/a) / g.
            # Integrating with u = u₀ cos(φ), f = 2Ω sin(φ):
            #     η(φ) = -(R Ω u₀ + u₀²/2) sin²(φ) / g
            # (equator high, poles depressed: water "piles up" against the centrifugal push)
            η_amplitude = -(R * Ω * u₀ + u₀^2 / 2) / g

            set!(model,
                 u = (λ, φ, z) -> u₀ * cosd(φ),
                 η = (λ, φ, z) -> η_amplitude * sind(φ)^2)

            simulation = Simulation(model; Δt = 100, stop_iteration = 10)
            run!(simulation)

            v_max = maximum(abs, interior(model.velocities.v))
            u_max = maximum(abs, interior(model.velocities.u))

            @test v_max < u₀ / 100   # should remain near zero
            @test u_max < 2 * u₀
            @test !any(isnan, interior(model.velocities.u))
            @test !any(isnan, interior(model.velocities.v))
        end

        #####
        ##### Non-curvilinear grids: metric terms must vanish (no curvature).
        ##### Includes ImmersedBoundaryGrid{<:RectilinearGrid} which previously
        ##### errored at `grid.radius` because the dispatch fell through to the
        ##### generic active branch.
        #####
        @testset "Zero metric on non-curvilinear grids [$(typeof(arch))]" begin
            rect = RectilinearGrid(arch; size=(8, 8, 4), extent=(1, 1, 1))
            ibg_rect = ImmersedBoundaryGrid(rect, GridFittedBottom((x, y) -> 0.5))

            advection = Centered()
            U = (zeros(8, 8, 4), zeros(8, 8, 4), zeros(8, 8, 4))
            V = (zeros(8, 8, 4), zeros(8, 8, 4), zeros(8, 8, 4))

            for grid in (rect, ibg_rect)
                @test U_dot_∇u_metric(1, 1, 1, grid, advection, U, V) == 0
                @test U_dot_∇v_metric(1, 1, 1, grid, advection, U, V) == 0
                @test U_dot_∇w_metric(1, 1, 1, grid, advection, U, V) == 0
            end
        end

        #####
        ##### Smoke test: NonhydrostaticModel + ImmersedBoundaryGrid wrapping a
        ##### RectilinearGrid must time-step without erroring on `grid.radius`.
        ##### This was the failing case before the dispatch was inverted to
        ##### default-zero / curvilinear-opt-in.
        #####
        @testset "NonhydrostaticModel + IBG{RectilinearGrid} [$(typeof(arch))]" begin
            rect = RectilinearGrid(arch; size=(8, 8, 8), extent=(1, 1, 1), halo=(6, 6, 6))
            ibg = ImmersedBoundaryGrid(rect, GridFittedBottom((x, y) -> 1//2))
            model = NonhydrostaticModel(ibg; advection=WENO())
            time_step!(model, 1//1000)
            @test !any(isnan, interior(model.velocities.u))
            @test !any(isnan, interior(model.velocities.v))
            @test !any(isnan, interior(model.velocities.w))
        end
    end
end
