using Oceananigans
using Oceananigans.Advection: U_dot_∇u_metric, U_dot_∇v_metric, U_dot_∇w_metric
using Test

#####
##### Solid body rotation tests for curvature metric terms
#####
##### Solid body rotation u = u₀ cos(φ), v = 0 is an exact steady-state
##### solution on the sphere. The Coriolis force and curvature metric terms
##### must balance exactly; any drift in v or w indicates incorrect metrics.
#####

@testset "Curvature metric terms" begin
    for arch in archs
        @testset "Solid body rotation [NonhydrostaticModel, $(typeof(arch))]" begin
            grid = LatitudeLongitudeGrid(arch;
                                         size = (36, 34, 4),
                                         halo = (4, 4, 4),
                                         longitude = (0, 360),
                                         latitude = (-80, 80),
                                         z = (-1000, 0))

            coriolis = HydrostaticSphericalCoriolis()

            model = NonhydrostaticModel(; grid, coriolis,
                                          advection = WENO(order=5))

            u₀ = 10.0
            set!(model, u = (λ, φ, z) -> u₀ * cosd(φ))

            Δt = 10.0
            simulation = Simulation(model; Δt, stop_iteration=10)
            run!(simulation)

            v_max = maximum(abs, interior(model.velocities.v))
            w_max = maximum(abs, interior(model.velocities.w))
            u_max = maximum(abs, interior(model.velocities.u))

            # v and w should remain small if metric terms correctly balance Coriolis
            @test v_max < 1.0   # should be ≪ u₀
            @test w_max < 1.0
            @test u_max < 2 * u₀  # u should not drift far from initial
            @test !any(isnan, interior(model.velocities.u))
            @test !any(isnan, interior(model.velocities.v))
            @test !any(isnan, interior(model.velocities.w))
        end

        @testset "Solid body rotation [HydrostaticFreeSurfaceModel, $(typeof(arch))]" begin
            grid = LatitudeLongitudeGrid(arch;
                                         size = (36, 34, 4),
                                         halo = (4, 4, 4),
                                         longitude = (0, 360),
                                         latitude = (-80, 80),
                                         z = (-1000, 0))

            free_surface = SplitExplicitFreeSurface(grid; substeps=10)
            coriolis = HydrostaticSphericalCoriolis()

            model = HydrostaticFreeSurfaceModel(; grid, coriolis, free_surface,
                                                  momentum_advection = WENOVectorInvariant(order=5))

            u₀ = 0.1
            g = model.free_surface.gravitational_acceleration
            R = grid.radius
            Ω = model.coriolis.rotation_rate

            # Balanced free surface for solid body rotation
            ηᵢ(λ, φ) = (R * Ω * u₀ + u₀^2 / 2) * sind(φ)^2 / g

            set!(model, u = (λ, φ, z) -> u₀ * cosd(φ), η = ηᵢ)

            Δt = 100.0
            simulation = Simulation(model; Δt, stop_iteration=10)
            run!(simulation)

            v_max = maximum(abs, interior(model.velocities.v))
            u_max = maximum(abs, interior(model.velocities.u))

            @test v_max < 0.01 * u₀  # should remain near zero
            @test u_max < 2 * u₀
            @test !any(isnan, interior(model.velocities.u))
            @test !any(isnan, interior(model.velocities.v))
        end

        @testset "Zero metric on RectilinearGrid [$(typeof(arch))]" begin
            grid = RectilinearGrid(arch; size=(8, 8, 4), extent=(1, 1, 1))

            # Metric terms should return zero on RectilinearGrid
            advection = Centered()
            U = (zeros(8, 8, 4), zeros(8, 8, 4), zeros(8, 8, 4))
            V = (zeros(8, 8, 4), zeros(8, 8, 4), zeros(8, 8, 4))

            @test U_dot_∇u_metric(1, 1, 1, grid, advection, U, V) == 0
            @test U_dot_∇v_metric(1, 1, 1, grid, advection, U, V) == 0
            @test U_dot_∇w_metric(1, 1, 1, grid, advection, U, V) == 0
        end
    end
end
