# Vertical-coordinate comparison

Validation experiments comparing three vertical coordinates in Oceananigans, after
Bruciaferri, Shapiro & Wobus (2018), *A multi-envelope vertical coordinate system for numerical ocean
modelling*, Ocean Dynamics 68:1239–1258.

The three coordinates (all conservation-tested to machine precision) are:

| Coordinate | Construction | Levels |
|---|---|---|
| **z-star** | `MutableVerticalDiscretization` (+ `GridFittedBottom` for topography) | geopotential, stepped bottom |
| **sigma** | `MultiEnvelopeVerticalDiscretization` + `LinearEnvelope` | terrain-following |
| **multi-envelope** | `MultiEnvelopeVerticalDiscretization` + `MultiEnvelope` | pycnocline-following surface, geopotential interior, bathymetry-following bottom |

## Scripts

Each writes a PNG next to itself. Run e.g. `julia --project validation/vertical_coordinates/<script>.jl`.

- **`grids_comparison.jl`** — draws the computational-level structure of the three coordinates over a
  shelf–slope bathymetry (`grids_comparison.png`). Fast; no time-stepping.

- **`dense_water_cascade.jl`** — CASC (§3.2.2): a dense patch released on a shelf cascades down a slope.
  Terrain-following coordinates let the plume descend along the levels rather than over z-steps. Output
  (`dense_water_cascade.png`): final dense-tracer cross-sections in physical space + plume-depth time series.

- **`cold_intermediate_layer.jl`** — CILF (§3.2.3): a passive tracer on a *doming* pycnocline. Geopotential
  levels cut across the sloped pycnocline (spurious vertical mixing); a pycnocline-following multi-envelope
  upper envelope keeps the tracer on its level. Output (`cold_intermediate_layer.png`): tracer cross-sections
  + vertical-spread time series (lower = less spurious diapycnal mixing).

- **`lock_release.jl`** — the original z-star lock-exchange conservation check.

All experiments use `vertical_coordinate = ZStarCoordinate()` (the free-surface-following time-stepper) and
`timestepper = :SplitRungeKutta3` (required for exact tracer conservation).
