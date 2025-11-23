## Averages, integrals, and cumulative integrals

Reduce first, ask questions later. This page is your brisk tour of how to squeeze juicy scalars and skinny profiles out of thick, three-dimensional `Field`s — without forgetting the geometry. We’ll start tiny and 2‑D, then get stretchier, add an immersed boundary, and wrap with where to go next.

We’ll use the same notational conventions as elsewhere:

- “Reductions” collapse one or more dimensions (average, integral, min/max, …).
- “Accumulations” compute cumulative scans along a dimension (e.g., vertical cumulative integral).
- Important: `mean` is a plain arithmetic mean over cells; `Average` is geometry‑aware and correct on stretched/curvy grids.

### Warm‑up: global reductions on a tiny 2‑D x–z grid

Let’s build a small x–z grid and a not‑too‑boring field.

```@example reductions
using Oceananigans
using Statistics
using CairoMakie
set_theme!(Theme(fontsize=18))
CairoMakie.activate!(type = "svg")

Nx, Nz = 8, 8
grid = RectilinearGrid(topology = (Periodic, Flat, Bounded),
                       size = (Nx, Nz),
                       x = (0, 2π),
                       z = (-1, 0))

c = CenterField(grid)
f(x, z) = (1 + sin(x)) * exp(z)
set!(c, f)
nothing
```

Global (over all dims) arithmetic reductions on raw data:

```@example reductions
global_mean    = mean(interior(c))
global_min_val = minimum(interior(c))
global_max_val = maximum(interior(c))
@info "Global mean/min/max (arithmetic)" global_mean global_min_val global_max_val
nothing
```

The same idea using Oceananigans reductions:

```@example reductions
max_field = compute!(Field(Reduction(maximum!, c; dims = :)))
min_field = compute!(Field(Reduction(minimum!, c; dims = :)))
avg_field = compute!(Field(Average(c; dims = :)))

@info "Global max via Reduction" max_field[1, 1, 1]
@info "Global min via Reduction" min_field[1, 1, 1]
@info "Global average via Average (geometry-aware)" avg_field[1, 1, 1]
nothing
```

### “dims” explained: reduce in x, then reduce in z

Global reductions are just a special case: `dims = :` means “all non‑Flat dimensions”.

Let’s reduce in x (dimension 1) to get a vertical profile, and reduce in z (dimension 3) to get a horizontal slice:

```@example reductions
cx_avg_over_x = compute!(Field(Average(c; dims = 1)))              # size (1, 1, Nz)
cx_max_over_z = compute!(Field(Reduction(maximum!, c; dims = 3)))  # size (Nx, 1, 1)

z_nodes = collect(znodes(c))
x_nodes = collect(xnodes(c))

fig = Figure(size = (700, 240))
ax1 = Axis(fig[1, 1], xlabel = "⟨c⟩ over x", ylabel = "z")
lines!(ax1, cx_avg_over_x[1, 1, :], z_nodes)
reverse!(ax1.scene.plots[1].attributes[:y]) # plot with z up

ax2 = Axis(fig[1, 2], xlabel = "x", ylabel = "max_z c")
lines!(ax2, x_nodes, cx_max_over_z[:, 1, 1])
fig
```

### When grids stretch, `mean` and `Average` part ways (and `Average` is right)

Arithmetic `mean` doesn’t know geometry. `Average` does: it multiplies by the correct metric (Δ, areas, volumes) before dividing by total measure. On uniform grids they coincide; on stretched grids they do not.

Let’s stretch z and compare a vertical mean with a vertical `Average` for the simple case `c(x, z) = z`:

```@example reductions
z_interfaces = [0.0, 0.02, 0.08, 0.20, 0.45, 0.70, 0.88, 1.00]                 # nonuniform spacing
grid_stretched = RectilinearGrid(topology = (Periodic, Flat, Bounded),
                                 size = (Nx, Nz),
                                 x = (0, 1),
                                 z = (first(z_interfaces), last(z_interfaces)))
grid_stretched = RectilinearGrid(size = (Nx, Nz),
                                 topology = (Periodic, Flat, Bounded),
                                 x = (0, 1),
                                 z = z_interfaces)

c_stretch = CenterField(grid_stretched)
set!(c_stretch, (x, z) -> z)

naive_mean_z = mean(interior(c_stretch); dims = 3)            # Nx×1 “plain” mean over centers
avg_mean_z   = compute!(Field(Average(c_stretch; dims = 3)))  # Nx×1 “metric-aware” mean

@info "Sample column: naive vs Average" naive_mean_z[1, 1] avg_mean_z[1, 1, 1]
nothing
```

On this `z ∈ [0, 1]` toy, the geometry‑aware mean is ≈ 0.5 regardless of spacing; the arithmetic mean of cell‑center samples generally isn’t.

### Immersed boundaries: wet‑only vertical integrals with a bottom

Immersed boundaries mask “inactive” cells (dry/inside the mountain). Integrals and averages can be restricted to the wet cells via the `condition` keyword of reductions.

We’ll build a gentle slope, set `u = 1` everywhere, and integrate over z. The column integral equals the wet thickness (it traces the bathymetry).

```@example reductions
using Oceananigans.ImmersedBoundaries: GridFittedBottom, inactive_cell

grid2 = RectilinearGrid(topology = (Bounded, Flat, Bounded),
                        size = (64, 48),
                        x = (0, 1),
                        z = (-1, 0))

bottom(x, y) = -0.2 - 0.3 * sin(2π * x)              # a wiggly bottom between -0.5 and -0.2
ibg = ImmersedBoundaryGrid(grid2, GridFittedBottom(bottom))

u = CenterField(ibg)
set!(u, 1)                                           # uniform 1 everywhere

wet_only(i, j, k, g, _) = !inactive_cell(i, j, k, g) # predicate for active (wet) cells

∫u_wet = compute!(Field(Integral(u; dims = 3, condition = wet_only)))  # x → wet thickness

fig = Figure(size = (700, 240))
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "∫ u dz (wet only)")
lines!(ax, collect(xnodes(∫u_wet)), ∫u_wet[:, 1, 1])
fig
```

Because `u = 1`, the integral is just the wet column height. If you had values in immersed cells (you shouldn’t!), the `condition` guards against counting them.

### What’s next: CumulativeIntegral and custom scans

Want the running integral up the water column? That’s `CumulativeIntegral`:

```@example reductions
grid1d = RectilinearGrid(topology = (Flat, Flat, Bounded),
                         size = 16,
                         z = (-1, 0))
c1d = CenterField(grid1d)
set!(c1d, z -> 1 + z)  # linear in z

Cz   = compute!(Field(CumulativeIntegral(c1d; dims = 3)))            # bottom → top
Czᵣ  = compute!(Field(CumulativeIntegral(c1d; dims = 3, reverse = true)))  # top → bottom

@info "Bottom-to-top end value equals Integral" Cz[1, 1, end]
@info "Top-to-bottom start value equals Integral" Czᵣ[1, 1, 1]
nothing
```

And for the DIY crowd: the `Scan` interface lets you plug in your own `reduce!` or `accumulate!` to build bespoke reductions and accumulations:

- `Reduction(maximum!, field; dims = ...)`
- `Accumulation(cumsum!, field; dims = ...)`

See also: the Operations tutorial for composing reductions of fancy `AbstractOperation`s (e.g. reduce a gradient magnitude).

### Bonus: spherical grids — Latitude‑Longitude and Tripolar

On curvy grids, `Integral` uses areas correctly. A classic check: integral of ones equals the spherical area of the domain.

Latitude–Longitude band:

```jldoctest reductions_latlon
using Oceananigans
using Statistics

grid = LatitudeLongitudeGrid(size = (36, 10, 3),
                             longitude = (0, 360),
                             latitude  = (-30, 30),
                             z = (-1000, 0))

ones_ccc = CenterField(grid); set!(ones_ccc, 1)
∫ones = compute!(Field(Integral(ones_ccc; dims = (1, 2))))

φ₀ = deg2rad(30.0)
expected = 4π * grid.radius^2 * sin(φ₀)          # area of spherical zone |φ| ≤ 30°
isapprox(∫ones[1, 1, 1], expected; rtol = 2e-2)

# output
true
```

Tripolar (set an explicit southernmost latitude and even longitudinal size):

```jldoctest reductions_tripolar
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid

grid = TripolarGrid(size = (32, 20, 2), southernmost_latitude = -80.0, z = (-1, 0))
ones_ccc = CenterField(grid); set!(ones_ccc, 1)
∫ones = compute!(Field(Integral(ones_ccc; dims = (1, 2))))

φsouth = deg2rad(-80.0)
expected = 2π * grid.radius^2 * (1 - sin(φsouth))  # area from φsouth to 90°
isapprox(∫ones[1, 1, 1], expected; rtol = 3e-2)

# output
true
```

### Quick summary

- Use plain `Statistics.mean` only for simple, uniform grids or for sanity checks.
- Prefer `Average`/`Integral` for correct, metric‑aware answers on stretched/curvy grids.
- Use `dims` to pick your reduction direction.
- `condition = ...` masks cells (e.g., wet‑only on immersed boundaries).
- `CumulativeIntegral` and `Scan`s unlock cumulative and custom reductions.


