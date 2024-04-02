using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: PartialCellBottom

using Printf
using GLMakie

underlying_grid = RectilinearGrid(CPU(),
                                  size = (128, 64), 
                                  halo=(4, 4), 
                                  y = (-1, 1),
                                  z = (-1, 0),
                                  topology=(Flat, Periodic, Bounded))

# A bump
h₀ = 0.5 # bump height
L = 0.25 # bump width
@inline h(y) = h₀ * exp(- y^2 / L^2)
@inline seamount(y) = - 1 + h(y)

minimum_fractional_cell_height = 0.2
immersed_boundaries = [
                       PartialCellBottom(seamount;
                                         minimum_fractional_cell_height),
                       GridFittedBottom(seamount)
                      ]

b = []
v = []

function progress(sim)
    vmax = maximum(abs, sim.model.velocities.v)
    @info @sprintf("Iter: %d, time: %.2e, max|v|: %.8e",
                   iteration(sim), time(sim), vmax)

    return nothing
end

for ib in immersed_boundaries
    grid = ImmersedBoundaryGrid(underlying_grid, ib)

    @show grid

    model = HydrostaticFreeSurfaceModel(; grid,
                                        tracer_advection = WENO(),
                                        momentum_advection = WENO(),
                                        coriolis = FPlane(f=1),
                                        tracers = :b,
                                        buoyancy = BuoyancyTracer())

    N² = 1
    bᵢ(y, z) = N² * z
    set!(model, u = 0, b = bᵢ)

    simulation = Simulation(model; Δt=1e-3, stop_iteration=3000)
    simulation.callbacks[:p] = Callback(progress, IterationInterval(100))

    run!(simulation)

    push!(b, Array(interior(model.tracers.b, 1, :, :)))
    push!(v, Array(interior(model.velocities.v, 1, :, :)))
end

b_partial = b[1]
b_full    = b[2]
Δb = b_full .- b_partial

v_partial = v[1]
v_full    = v[2]
Δv = v_full .- v_partial

fig = Figure(size=(1200, 1800))

#partial_cell_title = @sprintf("PartialCellBottom with ϵ = %.1f", minimum_fractional_Δz)
ax_bp = Axis(fig[1, 2], title="b PartialCellBottom")
ax_bf = Axis(fig[2, 2], title="b GridFittedBottom")
ax_bd = Axis(fig[3, 2], title="b Difference (GridFitted - PartialCell)")

color = (:black, 0.5)
linewidth = 3
levels = 15

hmbp = heatmap!(ax_bp, b_partial)
contour!(ax_bp, b_partial; levels, color, linewidth)
Colorbar(fig[1, 1], hmbp, label="Buoyancy", flipaxis=false)

hmbf = heatmap!(ax_bf, b_full)
contour!(ax_bf, b_full; levels, color, linewidth)
Colorbar(fig[2, 1], hmbf, label="Buoyancy", flipaxis=false)

hmbd = heatmap!(ax_bd, Δb)
Colorbar(fig[3, 1], hmbd, label="Buoyancy", flipaxis=false)

save("resting_bumpy_ocean_b.png", fig)

#display(fig)

fig = Figure(size=(1200, 1800))

ax_vp = Axis(fig[1, 2], title="v PartialCellBottom")
ax_vf = Axis(fig[2, 2], title="v GridFittedBottom")
ax_vd = Axis(fig[3, 2], title="v Difference (GridFitted - PartialCell)")

hmvp = heatmap!(ax_vp, v_partial)
contour!(ax_vp, v_partial; levels, color, linewidth)
Colorbar(fig[1, 1], hmvp, label="v", flipaxis=false)

hmvf = heatmap!(ax_vf, v_full)
contour!(ax_vf, v_full; levels, color, linewidth)
Colorbar(fig[2, 1], hmvf, label="v", flipaxis=false)

hmvd = heatmap!(ax_vd, Δv)
Colorbar(fig[3, 1], hmvd, label="v", flipaxis=false)

save("resting_bumpy_ocean_v.png", fig)

#display(fig)
