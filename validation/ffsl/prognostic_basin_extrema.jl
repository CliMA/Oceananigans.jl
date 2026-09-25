# Reproduce: julia --project validation/ffsl/prognostic_basin_extrema.jl

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: inactive_cell, MutableVerticalDiscretization
using Oceananigans.TimeSteppers: AdaptiveVerticallyImplicitDiscretization
using Printf
using Random
Random.seed!(11)

function basin_model(; zstar = false, vertical_scheme = WENO(order=5), free_surface_type = :split)
    Nx, Ny, Nz = 32, 32, 8
    z = zstar ? MutableVerticalDiscretization(collect(range(-1000, 0, length=Nz+1))) : (-1000, 0)
    underlying_grid = RectilinearGrid(size = (Nx, Ny, Nz), halo = (7, 7, 4), x = (0, 64kilometers), y = (0, 64kilometers), z = z,
                                      topology = (Bounded, Bounded, Bounded))
    ridge(x, y) = -1000 + 700 * exp(-(x - 40kilometers)^2 / 2(6kilometers)^2)
    island(x, y) = (x - 20kilometers)^2 + (y - 32kilometers)^2 < (6kilometers)^2 ? 10.0 : ridge(x, y)
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(island))

    free_surface = free_surface_type == :split ? SplitExplicitFreeSurface(grid; substeps = 60) : ImplicitFreeSurface()
    ffsl = FluxFormSemiLagrangian(; vertical_scheme)
    model = HydrostaticFreeSurfaceModel(grid; free_surface, coriolis = FPlane(f = 1e-4),
                                        buoyancy = BuoyancyTracer(), tracers = (:b, :c, :uniform, :cw),
                                        tracer_advection = (b = WENO(), c = ffsl, uniform = ffsl, cw = WENO()),
                                        momentum_advection = WENOVectorInvariant(order=5),
                                        vertical_coordinate = zstar ? ZStarCoordinate() : ZCoordinate(),
                                        timestepper = :SplitRungeKutta3)

    bᵢ(x, y, z) = 1e-5 * z + 2e-3 * tanh((y - 32kilometers) / 8kilometers)
    cᵢ(x, y, z) = rand()
    set!(model, b = bᵢ, c = cᵢ, uniform = 1, cw = cᵢ)
    return model
end

const cw₀_global = Ref{Any}()
function check(model; Nsteps = 100, Δt = 5minutes, label = "")
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    wet = [!inactive_cell(i, j, k, grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    ∫c = Field(Integral(model.tracers.c)); ∫w = Field(Integral(model.tracers.cw))
    compute!(∫c); compute!(∫w)
    C₀, W₀ = ∫c[1, 1, 1], ∫w[1, 1, 1]
    c₀ = interior(model.tracers.c)[wet]
    cw₀_global[] = interior(model.tracers.cw)[wet]
    maximum_uniform_error = 0.0
    maximum_courant = 0.0
    for n in 1:Nsteps
        time_step!(model, Δt)
        maximum_uniform_error = max(maximum_uniform_error, maximum(abs, interior(model.tracers.uniform)[wet] .- 1))
        u, v, _ = model.transport_velocities
        maximum_courant = max(maximum_courant, Δt * max(maximum(abs, interior(u)), maximum(abs, interior(v))) / 2kilometers)
    end
    compute!(∫c); compute!(∫w)
    c₁ = interior(model.tracers.c)[wet]
    cw₀ = cw₀_global[]
    cw₁ = interior(model.tracers.cw)[wet]
    @printf("   WENO tracer: min %.3e (was %.3e), max - 1 = %.3e (was %.3e)\n", minimum(cw₁), minimum(cw₀), maximum(cw₁) - 1, maximum(cw₀) - 1)
    @printf("%-28s max C = %.3f | uniform %.2e | mass FFSL %.2e WENO %.2e | NaN %s | min c %.3e (was %.3e) max c - 1 = %.3e (was %.3e)\n",
            label, maximum_courant, maximum_uniform_error, abs(∫c[1,1,1] - C₀) / C₀, abs(∫w[1,1,1] - W₀) / W₀, any(isnan, c₁),
            minimum(c₁), minimum(c₀), maximum(c₁) - 1, maximum(c₀) - 1)
end

for seed in 1:3
    Random.seed!(seed)
    check(basin_model(; free_surface_type = :implicit); label = "seed $seed static z, implicit, WENO5")
    Random.seed!(seed)
    check(basin_model(; free_surface_type = :implicit, vertical_scheme = UpwindBiased(order=1)); label = "seed $seed static z, implicit, Upwind1")
end
