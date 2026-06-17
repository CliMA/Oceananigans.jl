# # Dense-water cascading down a slope (CASC) — coordinate comparison
#
# After Bruciaferri et al. (2018), §3.2.2. A dense (negatively buoyant) patch released on a shelf cascades
# down a continental slope. The experiment is run on three vertical coordinates sharing the same bathymetry:
#
#   * z-star  — geopotential levels + stepped immersed bottom (the artefact-prone case).
#   * sigma   — terrain-following single bottom envelope.
#   * ME      — multi-envelope (geopotential interior + bathymetry-following bottom).
#
# Terrain-following coordinates (sigma, ME) let the plume descend along the levels instead of over z-steps.
# Produces an animation `dense_water_cascade.mp4` (dense-tracer cross-sections in physical space + the plume
# centre-of-mass depth normalised by its initial value) and a final-frame `dense_water_cascade.png`.
#
# Run: `julia --project validation/vertical_coordinates/dense_water_cascade.jl`.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Models: ZStarCoordinate
using Printf
using CairoMakie

const Nx, Ny, Nz = 96, 4, 24
const Lx, Lz = 1.5e5, 1000.0

slope(x, y=0) = 200 + 600 * clamp((x - 2e4) / 8e4, 0, 1)   # 200 m shelf → 800 m deep

function cascade_grid(coordinate)
    common = (size=(Nx, Ny, Nz), x=(0, Lx), y=(0, 1.2e4), halo=(4, 4, 4), topology=(Bounded, Periodic, Bounded))
    if coordinate === :zstar
        ug = RectilinearGrid(; common..., z=MutableVerticalDiscretization((-Lz, 0)))
        return ImmersedBoundaryGrid(ug, GridFittedBottom((x, y) -> -slope(x)))
    elseif coordinate === :sigma
        grid = RectilinearGrid(; common..., z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1)); formulation=LinearEnvelope()))
        materialize_envelopes!(grid, (x, y) -> slope(x))
        return grid
    elseif coordinate === :multienvelope
        grid = RectilinearGrid(; common..., z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1)); formulation=MultiEnvelope(level_counts=(14, 10))))
        materialize_envelopes!(grid, ((x, y) -> 500.0, (x, y) -> slope(x)))
        return grid
    end
end

function run_cascade(coordinate; stop_time=10days, Δt=120.0, save_interval=2hours)
    grid = cascade_grid(coordinate)
    # f = 0: a non-rotating gravity current cascades straight down the slope. With rotation and no bottom
    # friction the dense water geostrophically adjusts and flows *along* the slope instead of descending
    # (real rotating cascades need Ekman bottom drag — the Shapiro–Hill regime — which the paper includes).
    model = HydrostaticFreeSurfaceModel(grid;
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=30),
                                        tracers = (:b, :dense), buoyancy = BuoyancyTracer(),
                                        coriolis = nothing, momentum_advection = WENO(),
                                        tracer_advection = WENO(), timestepper = :SplitRungeKutta3,
                                        vertical_coordinate = ZStarCoordinate())
    set!(model, b=(x, y, z) -> 1e-5 * z - ((x < 2e4 && z > -180) ? 5e-3 : 0.0),
                dense=(x, y, z) -> (x < 2e4 && z > -180) ? 1.0 : 0.0)

    zc = [znode(i, 1, k, grid, Center(), Center(), Center()) for i in 1:Nx, k in 1:Nz]
    dense_weight() = dropdims(sum(Array(interior(model.tracers.dense)), dims=2), dims=2)

    snapshots = Matrix{Float64}[]; times = Float64[]; com = Float64[]
    function save(sim)
        push!(snapshots, Array(interior(sim.model.tracers.dense))[:, 1, :])
        push!(times, sim.model.clock.time / day)
        w = dense_weight(); push!(com, sum(w .* zc) / sum(w))
    end

    sim = Simulation(model; Δt, stop_time)
    sim.callbacks[:save] = Callback(save, TimeInterval(save_interval))
    save(sim)
    run!(sim)
    @info "$coordinate cascade done ($(length(times)) frames)"

    return (; grid, zc, snapshots, times, com)
end

coords = (:zstar, :sigma, :multienvelope)
labels = (zstar="z-star", sigma="sigma", multienvelope="multi-envelope")
results = Dict(c => run_cascade(c) for c in coords)

#####
##### Animation
#####

xf = xnodes(results[:sigma].grid, Face())
xc = xnodes(results[:sigma].grid, Center()) ./ 1e3
bathymetry = -slope.(xnodes(results[:sigma].grid, Center()))

# cell rectangles in physical (x, z) space for a coordinate (z varies per column)
function cell_rects(zc)
    zf = hcat(2zc[:, 1] .- zc[:, 2], (zc[:, 1:end-1] .+ zc[:, 2:end]) ./ 2, 2zc[:, end] .- zc[:, end-1])
    return [Rect2(xf[i] / 1e3, zf[i, k], (xf[i+1] - xf[i]) / 1e3, zf[i, k+1] - zf[i, k]) for i in 1:Nx, k in 1:Nz]
end

nframes = minimum(length(results[c].times) for c in coords)
frame = Observable(1)

fig = Figure(size=(1500, 760), fontsize=15)
for (n, c) in enumerate(coords)
    r = results[c]
    ax = Axis(fig[1, n]; title=labels[c], xlabel="x [km]", ylabel = n == 1 ? "z [m]" : "")
    rects = vec(cell_rects(r.zc))
    color = @lift vec(r.snapshots[$frame])
    poly!(ax, rects; color, colormap=:dense, colorrange=(0, 0.3), strokewidth=0)
    lines!(ax, xc, bathymetry, color=:black, linewidth=2)
    ylims!(ax, -Lz, 0)
end
Colorbar(fig[1, 4]; colormap=:dense, colorrange=(0, 0.3), label="dense-water tracer")

axd = Axis(fig[2, 1:3], title="Plume centre-of-mass depth, normalised by initial value (higher = more cascading)",
           xlabel="time [days]", ylabel="COM(t) / COM(0)")
for c in coords
    r = results[c]
    lines!(axd, r.times, r.com ./ r.com[1], linewidth=3, label=labels[c])
end
axislegend(axd, position=:lt)
tnow = @lift results[:sigma].times[$frame]
vlines!(axd, tnow, color=:gray, linestyle=:dash)
title = Label(fig[0, :], @lift(@sprintf("Dense-water cascade   t = %.1f days", results[:sigma].times[$frame])), fontsize=19)

record(fig, joinpath(@__DIR__, "dense_water_cascade.mp4"), 1:nframes; framerate=12) do i
    frame[] = i
end
frame[] = nframes
save(joinpath(@__DIR__, "dense_water_cascade.png"), fig)
@info "wrote dense_water_cascade.mp4 and .png"
