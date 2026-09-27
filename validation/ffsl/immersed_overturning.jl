# Reproduce: julia --project validation/ffsl/immersed_overturning.jl

using Oceananigans
include(joinpath(@__DIR__, "..", "..", "test", "setup", "volume_integrals.jl"))
using Oceananigans.Grids: inactive_cell
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Printf, Random

function run(vertical_scheme, Cx)
    Nx, Nz = 64, 16
    ug = RectilinearGrid(size = (Nx, Nz), halo = (7, 4), x = (0, 1), z = (-1, 0), topology = (Bounded, Flat, Bounded))
    grid = ImmersedBoundaryGrid(ug, GridFittedBottom(x -> -1 + 0.5 * exp(-(x - 0.7)^2 / 0.01)))
    Δx, Δz = 1 / Nx, 1 / Nz
    ψf(x, z) = sin(π * x) * sin(π * z) * (1 + 0.5 * sin(2π * x))
    ψ = zeros(Nx+1, Nz+1)
    for i in 1:Nx+1, k in 1:Nz+1
        dry = any(inactive_cell(ii, 1, kk, grid) for ii in (i-1, i), kk in (k-1, k))
        ψ[i, k] = dry ? 0.0 : ψf((i-1)Δx, -1 + (k-1)Δz)
    end
    u = XFaceField(grid); w = ZFaceField(grid)
    ui = [-(ψ[i, k+1] - ψ[i, k]) / Δz for i in 1:Nx+1, k in 1:Nz]
    wi = [(ψ[i+1, k] - ψ[i, k]) / Δx for i in 1:Nx, k in 1:Nz+1]
    interior(u)[:, 1, :] .= ui
    interior(w)[:, 1, :] .= wi
    fill_halo_regions!(u); fill_halo_regions!(w)
    Δt = Cx * Δx / maximum(abs, ui)
    model = HydrostaticFreeSurfaceModel(grid; velocities = PrescribedVelocityFields(; u, w), tracers = (:c, :uniform),
                                        tracer_advection = FluxFormSemiLagrangian(; vertical_scheme),
                                        timestepper = :SplitRungeKutta3, buoyancy = nothing)
    Random.seed!(3)
    set!(model, c = (x, z) -> exp(-((x - 0.3)^2 + (z + 0.3)^2) / 0.02) + 0.2 * rand(), uniform = 1)
    wet = [!inactive_cell(i, 1, k, grid) for i in 1:Nx, k in 1:Nz]
    c₀ = Array(interior(model.tracers.c))[:, 1, :][wet]
    ∫c₀ = volume_integral(model.tracers.c)
    err = 0.0
    for n in 1:100
        time_step!(model, Δt)
        err = max(err, maximum(abs, Array(interior(model.tracers.uniform))[:, 1, :][wet] .- 1))
    end
    c₁ = Array(interior(model.tracers.c))[:, 1, :][wet]
    @printf("%-8s Cx=%.1f Cz=%.2f uniform %.2e mass %.2e newmin %.3e newmax %.3e\n", first(summary(vertical_scheme), 6), Cx,
            Δt * maximum(abs, wi) / Δz, err, abs(volume_integral(model.tracers.c) - ∫c₀) / ∫c₀, minimum(c₁)-minimum(c₀), maximum(c₁)-maximum(c₀))
end

for s in (WENO(order=5), UpwindBiased(order=1)), C in (0.8, 1.5, 2.5)
    run(s, C)
end
