# Reproduce: julia --project validation/ffsl/zstar_channel.jl

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: inactive_cell, MutableVerticalDiscretization
using Printf, Random

# Shallow z-star channel with large, divergent flow so that the horizontal Courant number exceeds 1
function zstar_model(; immersed, Nx = 32, Ny = 16, Nz = 4, H = 10, substeps = 60)
    z = MutableVerticalDiscretization(collect(range(-H, 0, length=Nz+1)))
    ug = RectilinearGrid(size = (Nx, Ny, Nz), halo = (7, 7, 4), x = (0, 64kilometers), y = (0, 32kilometers), z = z,
                         topology = (Periodic, Bounded, Bounded))
    grid = immersed ? ImmersedBoundaryGrid(ug, GridFittedBottom((x, y) -> -H + 6 * exp(-((x - 40kilometers)^2 + (y - 16kilometers)^2) / (5kilometers)^2))) : ug
    model = HydrostaticFreeSurfaceModel(grid; free_surface = SplitExplicitFreeSurface(grid; substeps),
                                        momentum_advection = nothing, buoyancy = nothing,
                                        tracers = (:c, :uniform, :cw, :uniformw),
                                        tracer_advection = (c = FluxFormSemiLagrangian(), uniform = FluxFormSemiLagrangian(),
                                                            cw = WENO(), uniformw = WENO()),
                                        vertical_coordinate = ZStarCoordinate(), timestepper = :SplitRungeKutta3)
    Random.seed!(5)
    ηᵢ(x, y, z) = 1.5 * exp(-((x - 20kilometers)^2 + (y - 16kilometers)^2) / (6kilometers)^2)
    uᵢ(x, y, z) = 1.2 + 0.5 * sin(4π * y / 32kilometers)
    cᵢ(x, y, z) = exp(-((x - 32kilometers)^2 + (y - 16kilometers)^2) / (8kilometers)^2) + 0.1 * rand()
    set!(model, η = ηᵢ, u = uᵢ, c = cᵢ, cw = cᵢ, uniform = 1, uniformw = 1)
    return model
end

function run(model, Δt, Nsteps)
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    wet = [!inactive_cell(i, j, k, grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    ∫c = Field(Integral(model.tracers.c)); compute!(∫c); C₀ = ∫c[1, 1, 1]
    ∫w = Field(Integral(model.tracers.cw)); compute!(∫w); W₀ = ∫w[1, 1, 1]
    uniform_error = 0.0; uniform_error_weno = 0.0; courant = 0.0; ηrange = 0.0
    for n in 1:Nsteps
        time_step!(model, Δt)
        uniform_error = max(uniform_error, maximum(abs, interior(model.tracers.uniform)[wet] .- 1))
        uniform_error_weno = max(uniform_error_weno, maximum(abs, interior(model.tracers.uniformw)[wet] .- 1))
        u, v, _ = model.transport_velocities
        courant = max(courant, Δt * max(maximum(abs, interior(u)), maximum(abs, interior(v))) / 2kilometers)
        η = interior(model.free_surface.displacement)
        ηrange = max(ηrange, maximum(η) - minimum(η))
    end
    compute!(∫c); compute!(∫w)
    return (; courant, ηrange, uniform_error, uniform_error_weno,
              mass_error = abs(∫c[1, 1, 1] - C₀) / C₀, mass_error_weno = abs(∫w[1, 1, 1] - W₀) / W₀,
              finite = all(isfinite, interior(model.tracers.c)))
end

for immersed in (false, true), Δt in (600, 1800, 2400)
    r = run(zstar_model(; immersed), Δt, 30)
    @printf("immersed=%-5s Δt=%4d max C=%.2f η range=%.2f m | uniform FFSL %.2e WENO %.2e | ∫σc FFSL %.2e WENO %.2e | finite %s\n",
            immersed, Δt, r.courant, r.ηrange, r.uniform_error, r.uniform_error_weno, r.mass_error, r.mass_error_weno, r.finite)
end
