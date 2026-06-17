# # Hybrid "s-on-top-of-z-on-top-of-z" multi-envelope coordinate, with an immersed bottom
#
# A multi-envelope configuration that places the *variable* (terrain/pycnocline-following, "s") coordinate
# only in a surface zone, and keeps the deeper zones geopotential ("z"). Because σᵉ ≈ 1 in the geopotential
# zones, the reference and physical coordinates coincide there — so an immersed `GridFittedBottom` (whose
# mask lives in the reference coordinate) lands at the correct physical depth. This is the clean way to
# combine the multi-envelope coordinate with an immersed boundary for topography:
#
#   * surface zone (s): levels follow the doming pycnocline → low spurious diapycnal mixing.
#   * interior/deep zones (z): geopotential → low pressure-gradient error.
#   * bottom: an immersed boundary cuts the geopotential deep zone at the bathymetry.
#
# Produces `hybrid_s_on_z_on_z.png`. Run: `julia --project validation/vertical_coordinates/hybrid_s_on_z_on_z.jl`.

using Oceananigans
using Oceananigans.Grids: znode
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using CairoMakie

Nx, Nz = 80, 30
Lx, Lz = 2e5, 1500.0

pycnocline(x) = 120 + 130 * ((x - Lx/2) / (Lx/2))^2          # doming pycnocline, 120 → 250 m
bathymetry(x) = 500 + 700 * clamp((x - 2e4) / 1.4e5, 0, 1)   # shelf 500 m → 1200 m

# 3 zones (surface-first level counts): s (pycnocline) + transition + geopotential deep.
# e1 = pycnocline; e2, e3 flat ⇒ the deepest zone is geopotential (σᵉ = 1), where the immersed bottom sits.
level_counts = (10, 10, 10)
ug = RectilinearGrid(size=(Nx, 4, Nz), x=(0, Lx), y=(0, 2.5e4), halo=(4, 4, 4), topology=(Bounded, Periodic, Bounded),
                     z=MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, Nz+1)); formulation=MultiEnvelope(; level_counts)))
materialize_envelopes!(ug, ((x, y) -> pycnocline(x), (x, y) -> 1000.0, (x, y) -> Lz))
grid = ImmersedBoundaryGrid(ug, GridFittedBottom((x, y) -> -bathymetry(x)))

xc = xnodes(ug, Center()) ./ 1e3
ziface(g) = [znode(i, 1, k, g, Center(), Center(), Face()) for i in 1:Nx, k in 1:Nz+1]
z = ziface(ug)

fig = Figure(size=(1100, 620), fontsize=16)
ax = Axis(fig[1, 1], title="Multi-envelope hybrid: s (pycnocline) → z → z, with an immersed bottom",
          xlabel="x [km]", ylabel="z [m]")

# colour the level interfaces by zone (surface-first counts ⇒ top levels are the s zone)
zone_color(k) = k > Nz - level_counts[1] ? (:steelblue, "s: pycnocline-following") :
                k > Nz - level_counts[1] - level_counts[2] ? (:seagreen, "transition") :
                (:darkorange, "z: geopotential")
for k in 1:Nz+1
    lines!(ax, xc, z[:, k], color=zone_color(min(k, Nz))[1], linewidth=1)
end

band!(ax, xc, fill(-Lz, Nx), -bathymetry.(xnodes(ug, Center())), color=(:gray, 0.9))      # rock (immersed)
lines!(ax, xc, -bathymetry.(xnodes(ug, Center())), color=:black, linewidth=2.5, label="immersed bottom")
lines!(ax, xc, -pycnocline.(xnodes(ug, Center())), color=:red, linewidth=2.5, linestyle=:dash, label="pycnocline (upper envelope)")
ylims!(ax, -Lz, 30)

# legend for zones + features
elems = [LineElement(color=:steelblue), LineElement(color=:seagreen), LineElement(color=:darkorange),
         LineElement(color=:black, linewidth=2.5), LineElement(color=:red, linestyle=:dash, linewidth=2.5)]
labels = ["s: pycnocline-following", "transition", "z: geopotential", "immersed bottom", "pycnocline envelope"]
axislegend(ax, elems, labels, position=:rb, framevisible=true)

save(joinpath(@__DIR__, "hybrid_s_on_z_on_z.png"), fig)
@info "wrote hybrid_s_on_z_on_z.png"
