include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: inactive_cell, MutableVerticalDiscretization, Face, Center
using Oceananigans.Operators: Axᶠᶜᶜ, Ayᶜᶠᶜ, Δzᶠᶜᶜ, Δzᶜᶠᶜ, Vᶜᶜᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!

wet_cells(grid) = [!inactive_cell(i, j, k, grid) for i in 1:size(grid, 1), j in 1:size(grid, 2), k in 1:size(grid, 3)]
wet_values(c, wet) = Array(interior(c))[wet]

function integral(c)
    ∫c = Field(Integral(c))
    compute!(∫c)
    return Array(interior(∫c))[1, 1, 1]
end

# A shallow z-star channel with a strong divergent flow, so that the horizontal Courant number exceeds 1
# while the free surface varies by several percent of the depth
function zstar_channel_model(arch; immersed, H = 10)
    z = MutableVerticalDiscretization(collect(range(-H, 0, length=5)))
    underlying_grid = RectilinearGrid(arch; size = (32, 16, 4), halo = (7, 7, 4), x = (0, 64kilometers), y = (0, 32kilometers), z,
                                      topology = (Periodic, Bounded, Bounded))
    seamount(x, y) = -H + 6 * exp(-((x - 40kilometers)^2 + (y - 16kilometers)^2) / (5kilometers)^2)
    grid = immersed ? ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(seamount)) : underlying_grid

    ffsl = FluxFormSemiLagrangian()
    model = HydrostaticFreeSurfaceModel(grid; free_surface = SplitExplicitFreeSurface(grid; substeps = 60),
                                        momentum_advection = nothing, buoyancy = nothing,
                                        tracers = (:c, :uniform), tracer_advection = ffsl,
                                        vertical_coordinate = ZStarCoordinate(), timestepper = :SplitRungeKutta3)

    Random.seed!(5)
    ηᵢ(x, y, z) = 3/2 * exp(-((x - 20kilometers)^2 + (y - 16kilometers)^2) / (6kilometers)^2)
    uᵢ(x, y, z) = 1.2 + sin(4π * y / 32kilometers) / 2
    cᵢ(x, y, z) = exp(-((x - 32kilometers)^2 + (y - 16kilometers)^2) / (8kilometers)^2) + rand() / 10
    set!(model, η = ηᵢ, u = uᵢ, c = cᵢ, uniform = 1)
    return model
end

function tripolar_zstar_model(arch, fold_topology; H = 4)
    z = MutableVerticalDiscretization(collect(range(-H, 0, length=4)))
    underlying_grid = TripolarGrid(arch; size = (20, 32, 3), halo = (7, 7, 4), z, fold_topology)
    bump(λ, φ, λ₀) = exp(-(λ - λ₀)^2 / 50 - (φ - 55)^2 / 50)
    islands(λ, φ) = -H + 3H/2 * (bump(λ, φ, 70) + bump(λ, φ, 250) + bump(λ, φ, 430))
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(islands))

    model = HydrostaticFreeSurfaceModel(grid; free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                        momentum_advection = nothing, buoyancy = nothing,
                                        tracers = (:c, :uniform), tracer_advection = FluxFormSemiLagrangian(),
                                        vertical_coordinate = ZStarCoordinate(), timestepper = :SplitRungeKutta3)

    Random.seed!(2)
    # Rotation about an equatorial axis, which carries the tracers across the fold
    uᵢ(λ, φ, z) = 2 * sind(φ) * cosd(λ)
    vᵢ(λ, φ, z) = - 2 * sind(λ)
    ηᵢ(λ, φ, z) = exp(-((λ - 180)^2 + (φ - 60)^2) / 200)
    set!(model, u = uᵢ, v = vᵢ, η = ηᵢ, c = (λ, φ, z) -> 1 + cosd(φ) * cosd(λ) + rand() / 10, uniform = 1)
    return model
end

# Discretely non-divergent transport from a corner streamfunction: Ax u = - Δz δy ψ and Ay v = Δz δx ψ
function streamfunction_velocities(grid, ψ_function)
    ψ = Field{Face, Face, Center}(grid)
    set!(ψ, ψ_function)
    fill_halo_regions!(ψ)
    ψᶜᵖᵘ = on_architecture(CPU(), ψ)
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(grid)
    uᶜᵖᵘ = [- (ψᶜᵖᵘ[i, j+1, k] - ψᶜᵖᵘ[i, j, k]) * Δzᶠᶜᶜ(i, j, k, cpu_grid) / Axᶠᶜᶜ(i, j, k, cpu_grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    vᶜᵖᵘ = [(ψᶜᵖᵘ[i+1, j, k] - ψᶜᵖᵘ[i, j, k]) * Δzᶜᶠᶜ(i, j, k, cpu_grid) / Ayᶜᶠᶜ(i, j, k, cpu_grid) for i in 1:Nx, j in 1:Ny+1, k in 1:Nz]
    u = XFaceField(grid)
    v = YFaceField(grid)
    set!(u, uᶜᵖᵘ)
    set!(v, vᶜᵖᵘ)
    fill_halo_regions!((u, v))
    return u, v, uᶜᵖᵘ
end

@testset "FluxFormSemiLagrangian conservation with ZStarCoordinate" begin
    for arch in archs
        for immersed in (false, true)
            @testset "Uniform tracer and ∫σc in a z-star channel at C ≈ 2 (immersed = $immersed) [$(typeof(arch))]" begin
                @info "  Testing FluxFormSemiLagrangian with ZStarCoordinate at C ≈ 2 (immersed = $immersed) [$(typeof(arch))]..."
                model = zstar_channel_model(arch; immersed)
                wet = wet_cells(model.grid)
                C₀ = integral(model.tracers.c)
                Δt = 30minutes

                maximum_uniform_error = 0.0
                maximum_courant_number = 0.0
                free_surface_range = 0.0
                for _ in 1:20
                    time_step!(model, Δt)
                    maximum_uniform_error = max(maximum_uniform_error, maximum(abs, wet_values(model.tracers.uniform, wet) .- 1))
                    u, v, _ = model.transport_velocities
                    maximum_courant_number = max(maximum_courant_number, Δt * maximum(abs, interior(u)) / 2kilometers)
                    η = Array(interior(model.free_surface.displacement))
                    free_surface_range = max(free_surface_range, maximum(η) - minimum(η))
                end

                @info "    maximum Courant number $maximum_courant_number, uniform tracer error $maximum_uniform_error, " *
                      "∫σc relative error $(abs(integral(model.tracers.c) - C₀) / C₀), free surface range $free_surface_range"

                @test maximum_courant_number > 1.5
                @test free_surface_range > 0.1
                @test maximum_uniform_error < 1e-12
                @test abs(integral(model.tracers.c) - C₀) / C₀ < 1e-13
            end
        end

        for fold_topology in (RightCenterFolded, RightFaceFolded)
            @testset "Uniform tracer and ∫σc across the fold of a $fold_topology TripolarGrid [$(typeof(arch))]" begin
                @info "  Testing FluxFormSemiLagrangian with ZStarCoordinate on a $fold_topology TripolarGrid [$(typeof(arch))]..."
                model = tripolar_zstar_model(arch, fold_topology)
                wet = wet_cells(model.grid)
                C₀ = integral(model.tracers.c)

                maximum_uniform_error = 0.0
                for _ in 1:5
                    time_step!(model, 1hour)
                    maximum_uniform_error = max(maximum_uniform_error, maximum(abs, wet_values(model.tracers.uniform, wet) .- 1))
                end

                @test maximum_uniform_error < 1e-13
                @test abs(integral(model.tracers.c) - C₀) / C₀ < 1e-13
                @test all(isfinite, wet_values(model.tracers.c, wet))
            end
        end
    end
end

@testset "FluxFormSemiLagrangian on the polar rows of a LatitudeLongitudeGrid" begin
    for arch in archs
        @info "  Testing FluxFormSemiLagrangian on the polar rows of a LatitudeLongitudeGrid [$(typeof(arch))]..."
        grid = LatitudeLongitudeGrid(arch; size = (72, 30, 1), halo = (6, 6, 3), longitude = (0, 360), latitude = (-75, 75), z = (-100, 0))

        # Zonal rotation (the zonal Courant number grows as 1 / cos φ towards the polar rows) and a
        # rotation about an equatorial axis, tapered so that no volume crosses the walls at ±75ᵒ
        zonal_rotation(λ, φ, z) = 100kilometers * sind(φ)
        tilted_rotation(λ, φ, z) = 100kilometers * cosd(φ) * cosd(λ) * (1 - (sind(φ) / sind(75))^2)

        for ψ in (zonal_rotation, tilted_rotation)
            u, v, uᶜᵖᵘ = streamfunction_velocities(grid, ψ)
            cpu_grid = on_architecture(CPU(), grid)
            Nx, Ny, Nz = size(grid)
            # swept volume per unit time over the upstream cell volume
            unit_courant_x = maximum(abs(uᶜᵖᵘ[i, j, 1]) * Axᶠᶜᶜ(i, j, 1, cpu_grid) / Vᶜᶜᶜ(i, j, 1, cpu_grid) for i in 1:Nx, j in 1:Ny)
            Δt = 2.8 / unit_courant_x

            model = HydrostaticFreeSurfaceModel(grid; velocities = PrescribedVelocityFields(; u, v), tracers = (:c, :uniform),
                                                tracer_advection = FluxFormSemiLagrangian(), timestepper = :SplitRungeKutta3,
                                                buoyancy = nothing)

            # `set!(model, ...)` would reset the prescribed velocities on curvilinear grids, so the tracers are set directly
            Random.seed!(1)
            set!(model.tracers.c, (λ, φ, z) -> 1 + cosd(φ) * cosd(λ) / 2 + rand() / 10)
            set!(model.tracers.uniform, 1)
            fill_halo_regions!(model.tracers)

            V = [Vᶜᶜᶜ(i, j, k, cpu_grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
            mass(c) = sum(Array(interior(c)) .* V)
            M₀ = mass(model.tracers.c)
            c₀ = Array(interior(model.tracers.c))

            maximum_uniform_error = 0.0
            for _ in 1:20
                time_step!(model, Δt)
                maximum_uniform_error = max(maximum_uniform_error, maximum(abs, Array(interior(model.tracers.uniform)) .- 1))
            end
            c₁ = Array(interior(model.tracers.c))

            @test maximum_uniform_error < 1e-13
            @test abs(mass(model.tracers.c) - M₀) / M₀ < 1e-13
            @test minimum(c₁) ≥ minimum(c₀)
            @test maximum(c₁) ≤ maximum(c₀)
        end
    end
end
