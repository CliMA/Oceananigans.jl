# # Cold-intermediate-layer / pycnocline transport (CILF) — coordinate comparison
#
# After Bruciaferri et al. (2018), §3.2.3. A passive tracer is laid on a *doming* pycnocline and the flow is
# let adjust. Where the pycnocline is sloped, geopotential (z-star) levels cut across it, so advection
# numerically diffuses the tracer vertically off the pycnocline. A multi-envelope coordinate whose upper
# envelope follows the pycnocline keeps the tracer on its level. Three coordinates share the same flat-bottom
# basin and the same doming stratification:
#
#   * z-star  — geopotential levels (cross the doming pycnocline).
#   * sigma   — terrain-following the flat bottom ⇒ still ~geopotential near the surface.
#   * ME      — pycnocline-following upper envelope + geopotential interior.
#
# Produces an animation `cold_intermediate_layer.mp4` (tracer cross-sections + the tracer's vertical spread
# off the pycnocline, normalised by its initial value) and a final-frame `cold_intermediate_layer.png`.
# Run: `julia --project validation/vertical_coordinates/cold_intermediate_layer.jl`.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode
using Oceananigans.Models: ZStarCoordinate
using Printf
using CairoMakie

const Nx, Ny, Nz = 96, 4, 24
const Lx, Lz = 2e5, 1000.0

# doming pycnocline: 80 m deep at the centre, 200 m at the edges
pycnocline(x, y=0) = 80 + 120 * ((x - Lx/2) / (Lx/2))^2

function cil_grid(coordinate)
    common = (size=(Nx, Ny, Nz), x=(0, Lx), y=(0, 2.5e4), halo=(4, 4, 4), topology=(Bounded, Periodic, Bounded))
    if coordinate === :zstar
        return RectilinearGrid(; common..., z=MutableVerticalDiscretization((-Lz, 0)))
    elseif coordinate === :sigma
        grid = RectilinearGrid(; common..., z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1)); formulation=LinearEnvelope()))
        materialize_envelopes!(grid, (x, y) -> Lz)   # flat bottom ⇒ uniform σ
        return grid
    elseif coordinate === :multienvelope
        grid = RectilinearGrid(; common..., z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1)); formulation=MultiEnvelope(level_counts=(12, 12))))
        materialize_envelopes!(grid, ((x, y) -> pycnocline(x), (x, y) -> Lz); smooth_transitions=true)
        return grid
    end
end

function run_cil(coordinate; stop_time=8days, Δt=300.0, save_interval=4hours)
    grid = cil_grid(coordinate)
    model = HydrostaticFreeSurfaceModel(grid;
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=30),
                                        tracers = (:b, :c), buoyancy = BuoyancyTracer(),
                                        coriolis = FPlane(f=1e-4), momentum_advection = WENO(),
                                        tracer_advection = WENO(), timestepper = :SplitRungeKutta3,
                                        vertical_coordinate = ZStarCoordinate())
    set!(model, b=(x, y, z) -> z > -pycnocline(x) ? 0.0 : -2e-3,
                c=(x, y, z) -> abs(z + pycnocline(x)) < 30 ? 1.0 : 0.0)

    zc  = [znode(i, 1, k, grid, Center(), Center(), Center()) for i in 1:Nx, k in 1:Nz]
    pyc = [-pycnocline(xnodes(grid, Center())[i]) for i in 1:Nx, k in 1:Nz]
    cw() = dropdims(sum(Array(interior(model.tracers.c)), dims=2), dims=2)
    spread() = (w = cw(); sqrt(sum(w .* (zc .- pyc).^2) / sum(w)))   # RMS off-pycnocline distance

    snapshots = Matrix{Float64}[]; times = Float64[]; rms = Float64[]
    function save(sim)
        push!(snapshots, Array(interior(sim.model.tracers.c))[:, 1, :])
        push!(times, sim.model.clock.time / day); push!(rms, spread())
    end

    sim = Simulation(model; Δt, stop_time)
    sim.callbacks[:save] = Callback(save, TimeInterval(save_interval))
    save(sim)
    run!(sim)
    @info "$coordinate CILF done ($(length(times)) frames)"

    return (; grid, zc, snapshots, times, rms)
end

coords = (:zstar, :sigma, :multienvelope)
labels = (zstar="z-star", sigma="sigma", multienvelope="multi-envelope")
results = Dict(c => run_cil(c) for c in coords)

#####
##### Animation
#####

xf = xnodes(results[:zstar].grid, Face())
xc = xnodes(results[:zstar].grid, Center()) ./ 1e3
pyc_line = -pycnocline.(xnodes(results[:zstar].grid, Center()))

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
    poly!(ax, rects; color, colormap=:thermal, colorrange=(0, 1), strokewidth=0)
    lines!(ax, xc, pyc_line, color=:cyan, linewidth=2, linestyle=:dash)
    ylims!(ax, -400, 0)
end
Colorbar(fig[1, 4]; colormap=:thermal, colorrange=(0, 1), label="passive tracer")

axs = Axis(fig[2, 1:3], title="Tracer vertical spread off the pycnocline, normalised by initial value (lower = less spurious mixing)",
           xlabel="time [days]", ylabel="spread(t) / spread(0)")
for c in coords
    r = results[c]
    lines!(axs, r.times, r.rms ./ r.rms[1], linewidth=3, label=labels[c])
end
axislegend(axs, position=:lt)
tnow = @lift results[:zstar].times[$frame]
vlines!(axs, tnow, color=:gray, linestyle=:dash)
Label(fig[0, :], @lift(@sprintf("Cold-intermediate-layer transport   t = %.1f days", results[:zstar].times[$frame])), fontsize=19)

record(fig, joinpath(@__DIR__, "cold_intermediate_layer.mp4"), 1:nframes; framerate=12) do i
    frame[] = i
end
frame[] = nframes
save(joinpath(@__DIR__, "cold_intermediate_layer.png"), fig)
@info "wrote cold_intermediate_layer.mp4 and .png"
