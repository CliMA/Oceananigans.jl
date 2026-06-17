# # Horizontal pressure-gradient error over steep topography — coordinate comparison
#
# A continuously stratified ocean **at rest** over tall, steep topography. The true solution is no motion,
# so any current is a spurious horizontal-pressure-gradient error (HPGE). Three coordinates share the same
# bathymetry and the same realistic density profile (a smooth exponential thermocline via a linear equation
# of state — not a two-layer pycnocline):
#
#   * z-star — geopotential levels + stepped immersed bottom. Zero HPGE, but stair-steps topography.
#   * sigma  — single envelope: *every* level follows the topography, so even the upper-ocean levels tilt →
#              spurious currents through the whole water column.
#   * ME     — multi-envelope "z-on-top-of-s": a geopotential interior (flat ⇒ no HPGE) above the topography
#              + a terrain-following bottom zone. The interior stays clean like z-star.
#
# The topography peaks at ~400 m, *below* the 0–300 m geopotential interior, so the multi-envelope benefit is
# visible: in the upper ocean ME ≈ z-star (no spurious current) while sigma does not. (Where topography rises
# *into* the interior, ME degrades to sigma — geopotential interiors only help below their base.)
#
# Produces `steep_seamount_hpge.png`: the grids + the spurious-current field for each coordinate.
# Run: `julia --project validation/vertical_coordinates/steep_seamount_hpge.jl`.

using Oceananigans
using Oceananigans.Units
using Oceananigans.BuoyancyFormulations: SeawaterBuoyancy, LinearEquationOfState
using Oceananigans.Grids: znode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Models: ZStarCoordinate
using Printf
using CairoMakie

const Nx, Ny, Nz = 160, 4, 40
const Lx, Lz = 3e5, 2000.0
const interior_depth = 300.0   # base of the geopotential interior for ME

# tall steep topography peaking at ~400 m (below the geopotential interior) + a deeper ridge
bathymetry(x, y=0) = clamp(2000 - 1600 * exp(-((x - 1.2e5) / 1.5e4)^2) - 700 * exp(-((x - 2.1e5) / 3e4)^2), 400, 2000)

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion=2e-4), constant_salinity=35.0)
Tᵢ(x, y, z) = 4 + 16 * exp(z / 500)   # warm surface → cold deep, smooth exponential thermocline

function seamount_grid(coordinate)
    common = (size=(Nx, Ny, Nz), x=(0, Lx), y=(0, 2e4), halo=(4, 4, 4), topology=(Bounded, Periodic, Bounded))
    if coordinate === :zstar
        ug = RectilinearGrid(; common..., z=MutableVerticalDiscretization((-Lz, 0)))
        return ImmersedBoundaryGrid(ug, GridFittedBottom((x, y) -> -bathymetry(x)))
    elseif coordinate === :sigma
        grid = RectilinearGrid(; common..., z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1)); formulation=LinearEnvelope()))
        materialize_envelopes!(grid, (x, y) -> bathymetry(x))
        return grid
    elseif coordinate === :multienvelope
        # 6 geopotential levels (flat 300 m interior) + 34 terrain-following levels
        grid = RectilinearGrid(; common..., z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1)); formulation=MultiEnvelope(level_counts=(6, 34))))
        materialize_envelopes!(grid, shelf_safe_envelopes(bathymetry, (interior_depth,); minimum_thickness=20))
        return grid
    end
end

function run_seamount(coordinate; stop_time=1day, Δt=60.0)
    grid = seamount_grid(coordinate)
    model = HydrostaticFreeSurfaceModel(grid;
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=30),
                                        tracers = (:T,), buoyancy, coriolis = FPlane(f=1e-4),
                                        momentum_advection = WENO(), tracer_advection = WENO(),
                                        timestepper = :SplitRungeKutta3, vertical_coordinate = ZStarCoordinate())
    set!(model, T = Tᵢ)   # at rest

    sim = Simulation(model; Δt, stop_time)
    try
        run!(sim)
    catch err
        @warn "$coordinate blew up: $(sprint(showerror, err))"
    end

    speed = Array(interior(model.velocities.u))[1:Nx, 1, :]   # u is on x-faces (Nx+1); drop the last to match cells
    zc = [znode(i, 1, k, grid, Center(), Center(), Center()) for i in 1:Nx, k in 1:Nz]
    full_max = maximum(abs, filter(isfinite, speed))
    interior_mask = zc .> -(interior_depth - 50)
    upper_max = maximum(abs, filter(isfinite, speed[interior_mask]))
    @info "$coordinate: max|u| full = $(round(full_max; digits=3)), upper-ocean = $(round(upper_max; digits=4)) m/s"
    return (; grid, speed, zc, full_max, upper_max)
end

coords = (:zstar, :sigma, :multienvelope)
labels = (zstar="z-star", sigma="sigma (single envelope)", multienvelope="multi-envelope (z-on-s)")
results = Dict(c => run_seamount(c) for c in coords)

#####
##### Visualization: grids (top) + spurious-current field (bottom)
#####

xf = xnodes(results[:sigma].grid, Face())
xc = xnodes(results[:sigma].grid, Center()) ./ 1e3
bathy = -bathymetry.(xnodes(results[:sigma].grid, Center()))

function cell_rects(zc)
    zf = hcat(2zc[:, 1] .- zc[:, 2], (zc[:, 1:end-1] .+ zc[:, 2:end]) ./ 2, 2zc[:, end] .- zc[:, end-1])
    return [Rect2(xf[i] / 1e3, zf[i, k], (xf[i+1] - xf[i]) / 1e3, zf[i, k+1] - zf[i, k]) for i in 1:Nx, k in 1:Nz]
end

fig = Figure(size=(1500, 820), fontsize=14)
for (n, c) in enumerate(coords)
    r = results[c]
    ziface = [znode(i, 1, k, r.grid, Center(), Center(), Face()) for i in 1:Nx, k in 1:Nz+1]
    ax = Axis(fig[1, n]; title=labels[c], ylabel = n == 1 ? "z [m] — grid" : "")
    for k in 1:Nz+1
        lines!(ax, xc, ziface[:, k], color=:steelblue, linewidth=0.5)
    end
    band!(ax, xc, fill(-Lz, Nx), bathy, color=(:gray, 0.9)); lines!(ax, xc, bathy, color=:black, linewidth=1.5)
    hlines!(ax, -interior_depth, color=:green, linestyle=:dash); ylims!(ax, -Lz, 30)

    ax2 = Axis(fig[2, n]; xlabel="x [km]", ylabel = n == 1 ? "z [m] — spurious |u|" : "",
               title=@sprintf("max|u|: %.2f m/s (upper ocean %.1e)", r.full_max, r.upper_max))
    poly!(ax2, vec(cell_rects(r.zc)); color=vec(abs.(r.speed)), colormap=:amp, colorrange=(0, 0.5), strokewidth=0)
    band!(ax2, xc, fill(-Lz, Nx), bathy, color=(:gray, 0.9)); lines!(ax2, xc, bathy, color=:black, linewidth=1.5)
    hlines!(ax2, -interior_depth, color=:green, linestyle=:dash); ylims!(ax2, -Lz, 30)
end
Colorbar(fig[2, 4]; colormap=:amp, colorrange=(0, 0.5), label="spurious |u| [m s⁻¹]")
Label(fig[0, :], "HPGE at rest over steep topography (green dash = base of ME geopotential interior)", fontsize=17)

save(joinpath(@__DIR__, "steep_seamount_hpge.png"), fig)
@info "wrote steep_seamount_hpge.png"
