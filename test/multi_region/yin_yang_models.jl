include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "yin_yang_test_utils.jl"))

using Oceananigans: fields
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: FixedSubstepNumber

geographic_velocity(component) = (λ, φ, z) -> solid_body_velocity(λ, φ)[component]

# A free-surface bump in the overlap at (λ, φ) = (-90°, 0), which lies in both panels
bump(λ, φ, z) = exp(-((λ + 90)^2 + φ^2) / 400)

split_explicit_model(grid; kwargs...) =
    HydrostaticFreeSurfaceModel(grid; momentum_advection = VectorInvariant(), tracer_advection = nothing,
                                buoyancy = nothing, closure = nothing,
                                free_surface = SplitExplicitFreeSurface(substeps=10, extend_halos=false), kwargs...)

explicit_model(grid; kwargs...) =
    HydrostaticFreeSurfaceModel(grid; momentum_advection = VectorInvariant(), buoyancy = nothing, closure = nothing,
                                free_surface = ExplicitFreeSurface(), kwargs...)

# The largest difference between the z-star scaling σ = (H + η) / H at cell centers and its value from η
function zstar_scaling_error(model, H)
    grid = model.grid
    error = 0.0
    for r in 1:length(grid.region_grids)
        Nx, Ny, _ = size(getregion(grid, r))
        σ = on_architecture(CPU(), getregion(grid, r).z.σᶜᶜⁿ)[1:Nx, 1:Ny, 1]
        η = on_architecture(CPU(), interior(getregion(model.free_surface.displacement, r), :, :, 1))
        error = max(error, maximum(abs, σ .- (H .+ η) ./ H))
    end
    return error
end

@testset "Yin-Yang models" begin
    for arch in archs
        grid = YinYangGrid(arch; size=(48, 16, 2), z=(-1000, 0))

        @testset "Split-explicit free surface [$(typeof(arch))]" begin
            boundary_conditions = (; c = FieldBoundaryConditions(top = ValueBoundaryCondition((λ, φ, t) -> 2t)),
                                     d = FieldBoundaryConditions(top = ValueBoundaryCondition((λ, φ, t, b) -> b,
                                                                                            field_dependencies = :b)))
            model = split_explicit_model(grid; tracers = (:b, :c, :d), boundary_conditions)
            u, v, _ = model.velocities

            # set! takes geographic velocities and rotates them into each panel's basis
            set!(model, u = geographic_velocity(1), v = geographic_velocity(2))
            @test panel_error(u, panel_u(solid_body_velocity, grid); nodes = interior_nodes) < 1e-12
            @test panel_error(v, panel_v(solid_body_velocity, grid); nodes = interior_nodes) < 1e-12
            @test panel_error(u, panel_u(solid_body_velocity, grid)) < 2e-5
            @test panel_error(v, panel_v(solid_body_velocity, grid)) < 2e-5

            # A missing component is zero, and setting a tracer leaves the velocities untouched
            zonal_velocity(λ, φ) = (cosd(φ), zero(φ))
            set!(model, u = (λ, φ, z) -> cosd(φ))
            set!(model, b = 5)
            @test panel_error(u, panel_u(zonal_velocity, grid); nodes = interior_nodes) < 1e-12
            @test panel_error(v, panel_v(zonal_velocity, grid); nodes = interior_nodes) < 1e-12

            # Velocities in the panel bases are set as they are
            set!(model, u = 1, intrinsic_velocities = true)
            @test all(==(1), interior(getregion(u, 2)))

            # Without tracer advection, the top halos are set by time- and field-dependent conditions
            set!(model, u = 0, v = 0, η = bump, c = 1, d = 1)
            Δt = 600
            for _ in 1:3
                time_step!(model, Δt)
            end
            t = model.clock.time
            Nx, Ny, Nz = size(grid)
            for r in 1:2
                @test all(view(getregion(model.tracers.c, r).data, 1:Nx, 1:Ny, Nz+1) .≈ 2 * 2t - 1)
                @test all(view(getregion(model.tracers.d, r).data, 1:Nx, 1:Ny, Nz+1) .≈ 2 * 5 - 1)
            end

            # SplitRungeKutta3 gives the same wave in the interior of Yin
            rk3_model = split_explicit_model(grid; timestepper = :SplitRungeKutta3)
            set!(rk3_model, η = bump)
            for _ in 1:3
                time_step!(rk3_model, Δt)
            end
            η = rk3_model.free_surface.displacement
            ηᴬᴮ = model.free_surface.displacement
            @test 0.3 < maximum(interior(getregion(η, 1))) ≤ 1
            for r in 1:2
                @test interior(getregion(η, r)) ≈ interior(getregion(ηᴬᴮ, r)) rtol=1e-5
            end

            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; free_surface = SplitExplicitFreeSurface(substeps=10),
                                                                   momentum_advection = VectorInvariant())
        end

        mountain(λ, φ) = -1000 + 400 * exp(-(λ^2 + (φ - 30)^2) / 200)
        zstar_grid = YinYangGrid(arch; size=(48, 16, 2), z=MutableVerticalDiscretization((-1000, 0)), halo=(4, 4, 4))
        immersed_zstar_grid = ImmersedBoundaryGrid(zstar_grid, PartialCellBottom(mountain))

        @testset "Halos are consistent after a time step [$(typeof(arch)), $timestepper]" for (timestepper, step_grid) in
                ((:QuasiAdamsBashforth2, grid), (:SplitRungeKutta3, immersed_zstar_grid))

            # A refill right after `time_step!` must change nothing: no kernel writes to the halos after the step's fills
            model = explicit_model(step_grid; tracer_advection = WENO(order=5), coriolis = HydrostaticSphericalCoriolis(),
                                   tracers = :c, timestepper)
            set!(model, u = geographic_velocity(1), v = geographic_velocity(2), η = bump, c = (λ, φ, z) -> bump(λ + 60, φ, z))
            for _ in 1:3
                time_step!(model, 300)
            end

            u, v, _ = model.velocities
            c = model.tracers.c
            η = model.free_surface.displacement
            before = map(regional_parents, (u, v, c, η))
            fill_halo_regions!((u, v), model.clock, fields(model))
            fill_halo_regions!((c, η), model.clock, fields(model))
            @test map(regional_parents, (u, v, c, η)) == before

            # The halo columns get their vertical halos from the donor's, which the no-flux conditions set
            Nz = size(step_grid, 3)
            for field in (u, v, c), r in 1:2
                data = on_architecture(CPU(), getregion(field, r).data)
                @test [data[i, j, 0] for (i, j) in halo_nodes(step_grid)] == [data[i, j, 1] for (i, j) in halo_nodes(step_grid)]
                @test [data[i, j, Nz+1] for (i, j) in halo_nodes(step_grid)] == [data[i, j, Nz] for (i, j) in halo_nodes(step_grid)]
            end
        end

        @testset "Immersed z-star panels [$(typeof(arch))]" begin
            # The immersed boundaries of the panels move to another architecture as they are
            for r in 1:2
                panel = getregion(immersed_zstar_grid, r)
                cpu_panel = on_architecture(CPU(), panel)
                @test cpu_panel isa ImmersedBoundaryGrid
                @test parent(cpu_panel.immersed_boundary.bottom_height) == Array(parent(panel.immersed_boundary.bottom_height))
            end

            # The default free surface substeps without extending the halos
            default_model = HydrostaticFreeSurfaceModel(immersed_zstar_grid; momentum_advection = VectorInvariant(),
                                                        tracer_advection = nothing, buoyancy = nothing, closure = nothing)
            @test default_model.free_surface isa SplitExplicitFreeSurface
            @test default_model.free_surface.substepping isa FixedSubstepNumber
        end

        @testset "z-star on a partitioned latitude-longitude grid [$(typeof(arch))]" begin
            H = 1000
            underlying_grid = LatitudeLongitudeGrid(arch; size=(36, 18, 1), z=MutableVerticalDiscretization((-H, 0)),
                                                    longitude=(-180, 180), latitude=(-80, 80), radius=1e6)
            partitioned_grid = MultiRegionGrid(underlying_grid, partition = XPartition(2))
            model = explicit_model(partitioned_grid; tracers = (), timestepper = :SplitRungeKutta3)
            set!(model, η = (λ, φ, z) -> 100 * exp(-(λ^2 + φ^2) / 400))
            @test zstar_scaling_error(model, H) < 1e-12
            time_step!(model, 1)
            @test zstar_scaling_error(model, H) < 1e-12
        end
    end
end
