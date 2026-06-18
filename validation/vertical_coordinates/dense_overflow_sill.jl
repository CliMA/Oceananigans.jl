# # Dense overflow over a sill — immersed boundary + multi-envelope, z-star vs sigma vs ME
#
# A dense plume cascades down a continental slope and crosses a tall, **sharp** sill that rises *above* the
# multi-envelope's terrain-following bottom envelope. The smooth large-scale slope is represented by the
# coordinate (terrain-following), while the sharp sill crest is too steep for the envelope and is instead
# carved out by an **immersed boundary** — so this case exercises an immersed boundary and a multi-envelope
# grid *together*. The stratified ambient makes the interior pressure-gradient error matter as well. The three
# coordinates each fail differently:
#
#   * z-star — flat geopotential levels + stepped immersed bottom: the plume cascades down a staircase
#              (spurious step mixing) and the sill is a coarse staircase too.
#   * sigma  — single terrain-following envelope + immersed sill: smooth slope/sill, but every interior level
#              tilts, so the horizontal-pressure-gradient error stirs the stratified interior.
#   * ME     — multi-envelope (flat geopotential interior + terrain-following bottom) + immersed sill: smooth
#              descent like sigma, clean flat interior like z-star, sharp sill masked cleanly. Best of both.
#
# Produces `dense_overflow_sill.mp4` (dense-tracer cross-sections) + `.png`, and a plume centre-of-mass-depth
# time series (deeper = the plume cascaded further; staircases impede it).
#
# Run: `julia --project validation/vertical_coordinates/dense_overflow_sill.jl`.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Models: ZStarCoordinate
using Printf
using CairoMakie

const Nx, Nz = 160, 50         # genuinely 2D (x, z): Flat in y — no redundant y-columns
const Lx, Lz = 1.5e5, 2000.0
const interior_depth = 300.0   # base of the geopotential interior for ME
const N² = 1e-5                # weak (realistic deep) stratification so the dense water can cascade to the bottom
const Δb = 1.5e-2              # dense-anomaly buoyancy: neutral depth Δb/N² ≈ 1500 m, into the basin
const x_sill = 5e4

# Dense reservoir (600 m) draining over a sharp sill into a deep basin (1800 m). The reservoir is DEEPER than
# `interior_depth`, so the terrain-following bottom zone keeps healthy cells (a reservoir shallower than the
# interior collapses the bottom zone → thin-cell blow-up). The sill crest (~150 m) punches *through* the 300 m
# geopotential interior, so the immersed boundary masks it in BOTH the interior and the terrain-following zones.
large_scale(x, y=0) = 600 + 1200 * clamp((x - x_sill) / 1.5e4, 0, 1)   # 600 m reservoir → 1800 m basin
sill(x, y=0) = 450 * exp(-((x - x_sill) / 3e3)^2)                       # crest 600-450 = 150 m
bathymetry(x, y=0) = large_scale(x) - sill(x)
bottom_height(x, y=0) = -bathymetry(x)

function overflow_grid(coordinate)
    common = (size=(Nx, Nz), x=(0, Lx), halo=(4, 4), topology=(Bounded, Flat, Bounded))
    bottom = GridFittedBottom(bottom_height)
    if coordinate === :zstar
        ug = RectilinearGrid(; common..., z=MutableVerticalDiscretization((-Lz, 0)))
        return ImmersedBoundaryGrid(ug, bottom)
    elseif coordinate === :sigma
        ug = RectilinearGrid(; common..., z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1)); formulation=LinearEnvelope()))
        materialize_envelopes!(ug, large_scale)   # follow the large-scale shape; the sill punches above → immersed
        return ImmersedBoundaryGrid(ug, bottom)
    elseif coordinate === :multienvelope
        ug = RectilinearGrid(; common..., z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1)); formulation=MultiEnvelope(level_counts=(8, 42))))
        materialize_envelopes!(ug, shelf_safe_envelopes(large_scale, (interior_depth,); minimum_thickness=20))
        return ImmersedBoundaryGrid(ug, bottom)
    end
end

# the cascading gravity current accelerates to ~3 m/s over the sharp sill — that, not the internal-wave mode,
# sets the CFL here, so Δt = 60 s is needed (Δt = 120 s blows up over the sill).
function run_overflow(coordinate; stop_time=15days, Δt=60.0, save_interval=2hours)
    grid = overflow_grid(coordinate)
    model = HydrostaticFreeSurfaceModel(grid;
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=30),
                                        tracers = (:b, :dense), buoyancy = BuoyancyTracer(), coriolis = nothing,
                                        momentum_advection = WENO(), tracer_advection = WENO(),
                                        timestepper = :SplitRungeKutta3, vertical_coordinate = ZStarCoordinate())
    dense_patch(x, z) = x < x_sill - 5e3 ? 1.0 : 0.0   # fill the reservoir behind the sill (Flat y ⇒ (x,z))
    set!(model, b = (x, z) -> N² * z - Δb * dense_patch(x, z), dense = (x, z) -> dense_patch(x, z))

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
    @info "$coordinate done ($(length(times)) frames, final plume depth = $(round(com[end]; digits=1)) m)"
    return (; grid, zc, snapshots, times, com)
end

coords = (:zstar, :sigma, :multienvelope)
labels = (zstar="z-star", sigma="sigma + immersed", multienvelope="multi-envelope + immersed")
results = Dict(c => run_overflow(c) for c in coords)

#####
##### Animation
#####

xf = xnodes(results[:sigma].grid, Face())
xc = xnodes(results[:sigma].grid, Center()) ./ 1e3
bathy = -bathymetry.(xnodes(results[:sigma].grid, Center()))

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
    band!(ax, xc, fill(-Lz, Nx), bathy, color=(:gray, 0.9)); lines!(ax, xc, bathy, color=:black, linewidth=1.5)
    hlines!(ax, -interior_depth, color=:green, linestyle=:dash); ylims!(ax, -Lz, 0)
end
Colorbar(fig[1, 4]; colormap=:dense, colorrange=(0, 0.3), label="dense-water tracer")

axd = Axis(fig[2, 1:3], title="Plume centre-of-mass depth (deeper = cascaded further; staircases impede descent)",
           xlabel="time [days]", ylabel="depth [m]")
for c in coords
    r = results[c]
    lines!(axd, r.times, r.com, linewidth=3, label=labels[c])
end
axislegend(axd, position=:rt)
tnow = @lift results[:sigma].times[$frame]
vlines!(axd, tnow, color=:gray, linestyle=:dash)
Label(fig[0, :], @lift(@sprintf("Dense overflow over a sharp sill (immersed) on a terrain-following slope   t = %.1f days",
                                results[:sigma].times[$frame])), fontsize=18)

record(fig, joinpath(@__DIR__, "dense_overflow_sill.mp4"), 1:nframes; framerate=12) do i
    frame[] = i
end
frame[] = nframes
save(joinpath(@__DIR__, "dense_overflow_sill.png"), fig)
@info "wrote dense_overflow_sill.mp4 and .png"
