---
name: new-simulation
description: Set up and validate a new Oceananigans simulation, with or without a reference paper
user_invocable: true
---

# New Simulation

Set up, run, and visualize a new Oceananigans simulation.

## Step 1: Understand the Case

**If reproducing a paper:**
- Read the paper carefully and extract ALL parameters: domain size, resolution, physical constants,
  boundary conditions, initial conditions, forcing, closure parameters
- Check parameter tables (often "Table 1"), figure captions, and coordinate conventions

**If designing a new case:**
- Ask the user for the science goal or phenomenon to simulate
- Clarify: domain geometry, resolution, physics (buoyancy, Coriolis, closures), run duration
- Identify what quantities to diagnose and visualize

## Step 2: Set Up Geometry

- Create the grid and **visualize it immediately** (see Visualization below)
- Verify domain extents, topography/immersed boundaries, coordinate orientations
- If reproducing a paper, compare geometry to paper figures

## Step 3: Set Initial Conditions

- Apply initial conditions, then verify:
  - `minimum(field)` and `maximum(field)` make physical sense
  - Visualize spatial distribution
  - Dense water, stratification, velocity profiles are where they should be

## Step 4: Short Test Run

- Run a few timesteps on CPU at low resolution
- Check for NaNs: `maximum(abs, u)`, `maximum(abs, v)`, etc.
- Verify flow is developing (velocities changing from initial state)
- Check output files contain meaningful data

## Step 5: Progressive Validation

- Run a short simulation and visualize results
- Check physics: flow direction, velocity magnitude, mixing patterns
- If reproducing a paper, compare to early-time figures

## Step 6: Production Run and Comparison

- Run at full resolution / full duration
- Create diagnostic visualizations matching the science goal
- If reproducing a paper, match figure format, colormaps, axis ranges, time snapshots

## Visualization Guide

Use the Oceananigans Makie extension. **Plot `Field` objects directly** — avoid `interior()` and
explicit `nodes()` calls whenever possible. The extension handles coordinate extraction, GPU-to-CPU
transfer, and immersed boundary masking automatically.

### Setup

```julia
using CairoMakie
using Oceananigans
```

### Plotting Fields Directly

```julia
# 2D field (or 3D field with a Flat dimension) — just pass the field
fig, ax, plt = heatmap(field; colormap=:balance)

# Add to an existing axis
heatmap!(ax, field)

# 1D field
lines!(ax, field)
```

The extension auto-detects dimensionality, extracts grid coordinates, and sets axis labels.

### Slicing 3D Fields

3D fields cannot be plotted directly — you must slice them first. Use `view` to create
a 2D slice that remains a `Field`:

```julia
# Slice at k=1 (bottom), j=Ny÷2 (midpoint), etc.
b_surface = view(b, :, :, grid.Nz)
b_section = view(b, :, grid.Ny÷2, :)

heatmap!(ax, b_surface)
heatmap!(ax, b_section)
```

### Animations with Observables

```julia
using Oceananigans.OutputReaders: FieldTimeSeries

bt = FieldTimeSeries("output.jld2", "b")

n = Observable(1)
b_frame = @lift bt[$n]

fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, b_frame; colormap=:thermal)

record(fig, "animation.mp4", 1:length(bt.times)) do i
    n[] = i
end
```

### Spherical Grids

For `LatitudeLongitudeGrid`, `TripolarGrid`, etc., use `surface!` on an `Axis3`:

```julia
fig = Figure()
ax = Axis3(fig[1, 1]; aspect=:data)
surface!(ax, field; colormap=:viridis)
```

### Common Mistakes to Avoid

- **Don't use `interior(f)` to get data for plotting** — plot the `Field` directly
- **Don't call `nodes(grid, ...)` to build coordinate arrays** — the extension does this
- **Don't forget `compute!`** on `KernelComputedField` / computed fields before plotting
  (the extension calls `compute!` automatically, but be aware of it for debugging)
- **3D fields error** — always slice to 2D or 1D first

## Common Issues

- **NaN blowups**: timestep too large, unstable ICs, `if`/`else` on GPU (use `ifelse`)
- **Nothing happening**: wrong buoyancy sign, ICs not applied, forcing inactive
- **Wrong flow direction**: check coordinate conventions
- **GPU issues**: avoid branching, ensure type stability

## Output

- Place validation scripts in `validation/<case_name>/`
- Place example scripts in `examples/`
- Follow existing conventions in those directories
