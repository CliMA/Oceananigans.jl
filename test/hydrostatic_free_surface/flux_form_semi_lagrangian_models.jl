include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: inactive_cell
using Oceananigans.BoundaryConditions: fill_halo_regions!

wet_cells(grid) = [!inactive_cell(i, j, k, grid) for i in 1:size(grid, 1), j in 1:size(grid, 2), k in 1:size(grid, 3)]
wet_values(c, wet) = Array(interior(c))[wet]

# Flow around an island that pierces the surface of a closed basin. The streamfunction vanishes on the walls
# and on every corner of the island cells, so no volume crosses a wall or the coast.
function island_basin_model(arch; N = 32)
    Δ = 1 / N
    island_radius = 0.12
    island(x, y) = (x - 1/2)^2 + (y - 1/2)^2 < island_radius^2
    ψ(x, y) = 2 * max(0, sqrt((x - 1/2)^2 + (y - 1/2)^2) - island_radius - 3Δ/4) * sin(π * x) * sin(π * y)
    u(x, y, z, t) = - (ψ(x, y + Δ/2) - ψ(x, y - Δ/2)) / Δ
    v(x, y, z, t) = + (ψ(x + Δ/2, y) - ψ(x - Δ/2, y)) / Δ

    underlying_grid = RectilinearGrid(arch; size = (N, N, 2), halo = (7, 7, 4), x = (0, 1), y = (0, 1), z = (-1, 0),
                                      topology = (Bounded, Bounded, Bounded))
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> island(x, y) ? 1 : -1))

    return HydrostaticFreeSurfaceModel(grid; velocities = PrescribedVelocityFields(; u, v),
                                       tracers = (:c, :uniform), tracer_advection = FluxFormSemiLagrangian(),
                                       timestepper = :SplitRungeKutta3, buoyancy = nothing)
end

# Overturning cell in x-z over a ridge; the streamfunction vanishes next to every immersed cell
function overturning_model(arch; Nx = 32, Nz = 8, vertical_scheme = WENO(order=5))
    underlying_grid = RectilinearGrid(arch; size = (Nx, Nz), halo = (7, 4), x = (0, 1), z = (-1, 0),
                                      topology = (Bounded, Flat, Bounded))
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(x -> -1 + exp(-(x - 0.7)^2 / 0.01) / 2))

    Δx, Δz = 1 / Nx, 1 / Nz
    ψᶠᶠ(x, z) = sin(π * x) * sin(π * z) * (1 + sin(2π * x) / 2)
    ψ = zeros(Nx+1, Nz+1)
    for i in 1:Nx+1, k in 1:Nz+1
        dry = any(inactive_cell(i′, 1, k′, grid) for i′ in (i-1, i), k′ in (k-1, k))
        ψ[i, k] = dry ? 0 : ψᶠᶠ((i-1) * Δx, (k-1) * Δz - 1)
    end

    u = XFaceField(grid)
    w = ZFaceField(grid)
    set!(u, reshape([- (ψ[i, k+1] - ψ[i, k]) / Δz for i in 1:Nx+1, k in 1:Nz], Nx+1, 1, Nz))
    set!(w, reshape([(ψ[i+1, k] - ψ[i, k]) / Δx for i in 1:Nx, k in 1:Nz+1], Nx, 1, Nz+1))
    fill_halo_regions!(u)
    fill_halo_regions!(w)

    model = HydrostaticFreeSurfaceModel(grid; velocities = PrescribedVelocityFields(; u, w), tracers = (:c, :uniform),
                                        tracer_advection = FluxFormSemiLagrangian(; vertical_scheme),
                                        timestepper = :SplitRungeKutta3, buoyancy = nothing)

    return model, maximum(abs, Array(interior(u)))
end

function prognostic_basin_model(arch; vertical_scheme)
    underlying_grid = RectilinearGrid(arch; size = (16, 16, 4), halo = (7, 7, 4), x = (0, 64kilometers), y = (0, 64kilometers),
                                      z = (-1000, 0), topology = (Bounded, Bounded, Bounded))
    island(x, y) = (x - 20kilometers)^2 + (y - 32kilometers)^2 < (8kilometers)^2
    ridge(x, y) = -1000 + 700 * exp(-(x - 44kilometers)^2 / 2(6kilometers)^2)
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> island(x, y) ? 10 : ridge(x, y)))

    ffsl = FluxFormSemiLagrangian(; vertical_scheme)
    model = HydrostaticFreeSurfaceModel(grid; free_surface = ImplicitFreeSurface(), coriolis = FPlane(f = 1e-4),
                                        buoyancy = BuoyancyTracer(), tracers = (:b, :c, :uniform, :d),
                                        tracer_advection = (b = WENO(), c = ffsl, uniform = ffsl, d = WENO()),
                                        momentum_advection = WENOVectorInvariant(order=5),
                                        timestepper = :SplitRungeKutta3)

    bᵢ(x, y, z) = 1e-5 * z + 2e-3 * tanh((y - 32kilometers) / 8kilometers)
    set!(model, b = bᵢ, c = (x, y, z) -> rand(), uniform = 1, d = (x, y, z) -> rand())
    return model
end

@testset "FluxFormSemiLagrangian in HydrostaticFreeSurfaceModel" begin
    for arch in archs
        @testset "Island basin with prescribed flow at Courant number 2.5 [$(typeof(arch))]" begin
            @info "  Testing FluxFormSemiLagrangian in an immersed island basin at C = 2.5 [$(typeof(arch))]..."
            model = island_basin_model(arch)
            N = size(model.grid, 1)
            u, v, _ = model.velocities
            maximum_velocity = max(maximum(abs, interior(Field(u))), maximum(abs, interior(Field(v))))
            Δt = 2.5 / (N * maximum_velocity)

            set!(model, c = (x, y, z) -> exp(-((x - 0.2)^2 + (y - 0.5)^2) / 0.01) + rand() / 10, uniform = 1)
            wet = wet_cells(model.grid)
            c₀ = wet_values(model.tracers.c, wet)

            maximum_uniform_error = 0.0
            for _ in 1:40
                time_step!(model, Δt)
                maximum_uniform_error = max(maximum_uniform_error, maximum(abs, wet_values(model.tracers.uniform, wet) .- 1))
            end
            c₁ = wet_values(model.tracers.c, wet)

            @test maximum_uniform_error < 1e-12
            @test abs(sum(c₁) - sum(c₀)) / sum(c₀) < 1e-14
            @test all(isfinite, c₁)
            @test minimum(c₁) ≥ minimum(c₀) - 1e-14
            @test maximum(c₁) ≤ maximum(c₀) + 1e-14
        end

        @testset "Overturning over a ridge at horizontal Courant number 1.5 [$(typeof(arch))]" begin
            @info "  Testing FluxFormSemiLagrangian with vertical WENO in an overturning cell [$(typeof(arch))]..."
            model, maximum_velocity = overturning_model(arch)
            Δt = 1.5 / (size(model.grid, 1) * maximum_velocity)
            Random.seed!(3)
            set!(model, c = (x, z) -> exp(-((x - 0.3)^2 + (z + 0.3)^2) / 0.02) + rand() / 5, uniform = 1)
            wet = wet_cells(model.grid)
            c₀ = wet_values(model.tracers.c, wet)

            maximum_uniform_error = 0.0
            for _ in 1:50
                time_step!(model, Δt)
                maximum_uniform_error = max(maximum_uniform_error, maximum(abs, wet_values(model.tracers.uniform, wet) .- 1))
            end
            c₁ = wet_values(model.tracers.c, wet)

            @test maximum_uniform_error < 1e-13
            @test abs(sum(c₁) - sum(c₀)) / sum(c₀) < 1e-14
            @test minimum(c₁) ≥ minimum(c₀)
            @test maximum(c₁) ≤ maximum(c₀)
        end

        @testset "Prognostic basin mixing FFSL and WENO tracers [$(typeof(arch))]" begin
            @info "  Testing FluxFormSemiLagrangian in a prognostic immersed basin [$(typeof(arch))]..."
            # A monotone vertical scheme isolates the extrema created by the horizontal step (none expected);
            # WENO(order=5) in the vertical is not monotone and does create small extrema, for FFSL and WENO tracers alike.
            for vertical_scheme in (UpwindBiased(order=1), WENO(order=5))
                Random.seed!(1)
                model = prognostic_basin_model(arch; vertical_scheme)
                @test model.advection.c isa FluxFormSemiLagrangian
                @test model.advection.d isa WENO
                wet = wet_cells(model.grid)
                c₀ = wet_values(model.tracers.c, wet)

                maximum_uniform_error = 0.0
                for _ in 1:30
                    time_step!(model, 5minutes)
                    maximum_uniform_error = max(maximum_uniform_error, maximum(abs, wet_values(model.tracers.uniform, wet) .- 1))
                end
                c₁ = wet_values(model.tracers.c, wet)

                @test maximum_uniform_error < 1e-12
                @test all(isfinite, c₁)
                @test all(isfinite, wet_values(model.tracers.d, wet))

                if vertical_scheme isa UpwindBiased
                    @test minimum(c₁) ≥ minimum(c₀)
                    @test maximum(c₁) ≤ maximum(c₀)
                end
            end
        end
    end
end
