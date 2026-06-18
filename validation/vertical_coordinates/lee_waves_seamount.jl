# # Dynamical flow over a seamount — z-star vs sigma vs multi-envelope (z-on-s)
#
# A stratified mean flow past a tall, steep seamount radiates internal lee waves and forms a wake. Unlike the
# resting `steep_seamount_hpge.jl` case (which measures the *static* pressure-gradient error), this is a fully
# **dynamical** experiment: the true response lives both at the bottom (where the flow is deflected) and in the
# interior (where waves radiate upward). The three vertical coordinates corrupt it in complementary ways:
#
#   * z-star — flat geopotential interior (clean wave propagation) BUT a stair-stepped seamount scatters the
#              near-bottom flow into grid-scale noise.
#   * sigma  — smooth terrain-following seamount (clean generation) BUT *every* interior level tilts, so the
#              horizontal-pressure-gradient error injects spurious currents through the whole water column.
#   * ME     — multi-envelope "z-on-s": a flat geopotential interior (clean, like z-star) sitting ON TOP OF a
#              terrain-following bottom (smooth, like sigma). It should be the cleanest of the three.
#
# The seamount peak (~600 m) sits *below* the base of the ME geopotential interior (400 m), so the multi-envelope
# benefit is visible: its upper ocean stays flat like z-star while the bottom follows the topography like sigma.
#
# Produces `lee_waves_seamount.mp4` (vertical-velocity field for the three coordinates) + a final-frame `.png`,
# and a time series of interior kinetic energy (the spurious-flow proxy: lower is cleaner).
#
# Run: `julia --project validation/vertical_coordinates/lee_waves_seamount.jl`.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Models: ZStarCoordinate
using Printf
using CairoMakie

const Nx, Ny, Nz = 200, 4, 50
const Lx, Lz = 3e5, 2000.0
const interior_depth = 400.0   # base of the geopotential interior for ME
const U₀ = 0.1                 # mean flow [m s⁻¹]
const N² = 1e-4                # background stratification [s⁻²]

# a tall steep seamount peaking at ~600 m (below the 400 m geopotential interior), isolated in a periodic channel
bathymetry(x, y=0) = clamp(2000 - 1400 * exp(-((x - Lx/2) / 2.0e4)^2), 600, 2000)

function seamount_grid(coordinate)
    common = (size=(Nx, Ny, Nz), x=(0, Lx), y=(0, 2e4), halo=(4, 4, 4), topology=(Periodic, Periodic, Bounded))
    if coordinate === :zstar
        ug = RectilinearGrid(; common..., z=MutableVerticalDiscretization((-Lz, 0)))
        return ImmersedBoundaryGrid(ug, GridFittedBottom((x, y) -> -bathymetry(x)))
    elseif coordinate === :sigma
        grid = RectilinearGrid(; common..., z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1)); formulation=LinearEnvelope()))
        materialize_envelopes!(grid, (x, y) -> bathymetry(x))
        return grid
    elseif coordinate === :multienvelope
        grid = RectilinearGrid(; common..., z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1)); formulation=MultiEnvelope(level_counts=(10, 40))))
        materialize_envelopes!(grid, shelf_safe_envelopes(bathymetry, (interior_depth,); minimum_thickness=20))
        return grid
    end
end

function run_seamount(coordinate; stop_time=2days, Δt=100.0, save_interval=30minutes)
    grid = seamount_grid(coordinate)
    model = HydrostaticFreeSurfaceModel(grid;
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=30),
                                        tracers = (:b,), buoyancy = BuoyancyTracer(), coriolis = nothing,
                                        momentum_advection = WENO(), tracer_advection = WENO(),
                                        timestepper = :SplitRungeKutta3, vertical_coordinate = ZStarCoordinate())
    set!(model, u = U₀, b = (x, y, z) -> N² * z)

    zc = [znode(i, 1, k, grid, Center(), Center(), Center()) for i in 1:Nx, k in 1:Nz]

    snapshots = Matrix{Float64}[]; times = Float64[]; noise = Float64[]
    function save(sim)
        w = Array(interior(sim.model.velocities.w))[:, 1, :]
        wc = (w[:, 1:Nz] .+ w[:, 2:Nz+1]) ./ 2   # w to centres for plotting on the cell grid
        push!(snapshots, wc)
        push!(times, sim.model.clock.time / day)
        # grid-scale (2Δx) noise in w isolates the coordinate artefact (stair-step scattering / HPGE checkerboard)
        # from the smooth physical lee waves; circshift handles the periodic x-boundary.
        hp = wc .- (circshift(wc, (1, 0)) .+ circshift(wc, (-1, 0))) ./ 2
        push!(noise, sqrt(sum(hp .^ 2) / length(hp)))
    end

    sim = Simulation(model; Δt, stop_time)
    sim.callbacks[:save] = Callback(save, TimeInterval(save_interval))
    save(sim)
    run!(sim)
    @info "$coordinate done ($(length(times)) frames, final grid-scale w-noise = $(round(noise[end]; sigdigits=3)))"
    return (; grid, zc, snapshots, times, noise)
end

coords = (:zstar, :sigma, :multienvelope)
labels = (zstar="z-star", sigma="sigma", multienvelope="multi-envelope (z-on-s)")
results = Dict(c => run_seamount(c) for c in coords)

#####
##### Visualization: vertical-velocity field (animation) + interior-KE time series
#####

xf = collect(range(0, Lx, Nx+1))   # periodic-x: xnodes(Face) has only Nx points, so build the Nx+1 cell edges
xc = xnodes(results[:sigma].grid, Center()) ./ 1e3
bathy = -bathymetry.(xnodes(results[:sigma].grid, Center()))

function cell_rects(zc)
    zf = hcat(2zc[:, 1] .- zc[:, 2], (zc[:, 1:end-1] .+ zc[:, 2:end]) ./ 2, 2zc[:, end] .- zc[:, end-1])
    return [Rect2(xf[i] / 1e3, zf[i, k], (xf[i+1] - xf[i]) / 1e3, zf[i, k+1] - zf[i, k]) for i in 1:Nx, k in 1:Nz]
end

nframes = minimum(length(results[c].times) for c in coords)
frame = Observable(1)
wlim = 0.02

fig = Figure(size=(1500, 1120), fontsize=15)

# Row 1 — the grids: the structural reason z-on-s is a compromise (z-star flat+stepped, sigma all-tilted,
# multi-envelope flat geopotential interior above a smooth terrain-following bottom).
for (n, c) in enumerate(coords)
    r = results[c]
    ziface = [znode(i, 1, k, r.grid, Center(), Center(), Face()) for i in 1:Nx, k in 1:Nz+1]
    ax = Axis(fig[1, n]; title=labels[c], ylabel = n == 1 ? "z [m] — grid" : "")
    for k in 1:Nz+1
        lines!(ax, xc, ziface[:, k], color=:steelblue, linewidth=0.4)
    end
    band!(ax, xc, fill(-Lz, Nx), bathy, color=(:gray, 0.9)); lines!(ax, xc, bathy, color=:black, linewidth=1.5)
    hlines!(ax, -interior_depth, color=:green, linestyle=:dash); ylims!(ax, -Lz, 0)
end

# Row 2 — the dynamical vertical-velocity field
for (n, c) in enumerate(coords)
    r = results[c]
    ax = Axis(fig[2, n]; xlabel="x [km]", ylabel = n == 1 ? "z [m] — w" : "")
    rects = vec(cell_rects(r.zc))
    color = @lift vec(r.snapshots[$frame])
    poly!(ax, rects; color, colormap=:balance, colorrange=(-wlim, wlim), strokewidth=0)
    band!(ax, xc, fill(-Lz, Nx), bathy, color=(:gray, 0.9)); lines!(ax, xc, bathy, color=:black, linewidth=1.5)
    hlines!(ax, -interior_depth, color=:green, linestyle=:dash); ylims!(ax, -Lz, 0)
end
Colorbar(fig[2, 4]; colormap=:balance, colorrange=(-wlim, wlim), label="vertical velocity w [m s⁻¹]")

axk = Axis(fig[3, 1:3], title="Grid-scale (2Δx) noise in w — coordinate artefact proxy, lower = cleaner",
           xlabel="time [days]", ylabel="RMS grid-scale w [m s⁻¹]")
for c in coords
    r = results[c]
    lines!(axk, r.times, r.noise, linewidth=3, label=labels[c])
end
axislegend(axk, position=:lt)
tnow = @lift results[:sigma].times[$frame]
vlines!(axk, tnow, color=:gray, linestyle=:dash)
Label(fig[0, :], @lift(@sprintf("Stratified flow over a seamount   t = %.2f days   (green dash = base of ME interior)",
                                results[:sigma].times[$frame])), fontsize=18)

record(fig, joinpath(@__DIR__, "lee_waves_seamount.mp4"), 1:nframes; framerate=12) do i
    frame[] = i
end
frame[] = nframes
save(joinpath(@__DIR__, "lee_waves_seamount.png"), fig)
@info "wrote lee_waves_seamount.mp4 and .png"
