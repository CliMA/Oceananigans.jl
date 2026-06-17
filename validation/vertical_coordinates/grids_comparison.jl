# # Vertical-coordinate comparison: grid structure over a slope
#
# Draws the computational-level structure of three vertical coordinates over the same shelf–slope
# bathymetry, illustrating how each represents topography:
#
#   * z-star  — geopotential levels riding the free surface; topography is a *stepped* immersed bottom.
#   * sigma   — a single bottom envelope (`LinearEnvelope`): levels follow the bathymetry smoothly.
#   * ME      — multi-envelope (`MultiEnvelope`): a pycnocline-following upper envelope, a geopotential
#               interior, and a bathymetry-following bottom.
#
# Run: `julia --project validation/vertical_coordinates/grids_comparison.jl` → `grids_comparison.png`.

using Oceananigans
using Oceananigans.Grids: znode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using CairoMakie

Nx, Nz = 100, 20
Lx, Lz = 1.5e5, 1000.0

# shelf (200 m) → continental slope → deep (800 m)
slope(x) = 200 + 600 * clamp((x - 2e4) / 8e4, 0, 1)

# z-star: geopotential levels + stepped immersed bottom
zstar = ImmersedBoundaryGrid(RectilinearGrid(size=(Nx, Nz), x=(0, Lx),
                                             z=MutableVerticalDiscretization((-Lz, 0)),
                                             topology=(Bounded, Flat, Bounded)),
                             GridFittedBottom(x -> -slope(x)))

# sigma: single terrain-following bottom envelope
sigma = RectilinearGrid(size=(Nx, Nz), x=(0, Lx),
                        z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1)); formulation=LinearEnvelope()),
                        topology=(Bounded, Flat, Bounded))
materialize_envelopes!(sigma, slope)

# multi-envelope: pycnocline (150 m) + geopotential interior + bathymetry-following bottom
multienvelope = RectilinearGrid(size=(Nx, Nz), x=(0, Lx),
                                z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1));
                                                                      formulation=MultiEnvelope(level_counts=(10, 10))),
                                topology=(Bounded, Flat, Bounded))
materialize_envelopes!(multienvelope, (x -> 150.0, slope))

xc = xnodes(sigma, Center()) ./ 1e3   # km
interface_depths(grid) = [znode(i, 1, k, grid, Center(), Center(), Face()) for i in 1:Nx, k in 1:Nz+1]

fig = Figure(size=(1500, 420), fontsize=15)
grids  = (zstar, sigma, multienvelope)
titles = ("z-star (geopotential + stepped bottom)", "sigma (terrain-following)", "multi-envelope")

for (n, (grid, title)) in enumerate(zip(grids, titles))
    ax = Axis(fig[1, n]; title, xlabel="x [km]", ylabel = n == 1 ? "z [m]" : "")
    bathymetry = -slope.(xnodes(sigma, Center()))
    z = interface_depths(grid)
    for k in 1:Nz+1
        lines!(ax, xc, z[:, k], color=:steelblue, linewidth=0.8)                          # computational levels
    end
    band!(ax, xc, fill(-Lz, Nx), bathymetry, color=(:gray, 0.85))                         # rock (hides inactive levels)
    lines!(ax, xc, bathymetry, color=:black, linewidth=2)                                 # bathymetry
    ylims!(ax, -Lz, 30)
end

save(joinpath(@__DIR__, "grids_comparison.png"), fig)
@info "wrote grids_comparison.png"
