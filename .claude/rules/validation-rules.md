---
paths:
  - validation/**/*
---

# Validation Rules

When implementing a simulation from a published paper:

## Workflow

1. **Extract ALL parameters** from the paper: domain size, resolution, physical constants,
   boundary conditions, initial conditions, forcing, closure parameters.
   Check parameter tables, figure captions, and coordinate conventions.

2. **Verify geometry BEFORE running** - visualize the grid/domain, check domain extents,
   topography, and coordinate orientations match the paper.

3. **Verify initial conditions** - check `minimum(field)` / `maximum(field)`, visualize
   spatial distribution, confirm stratification and density placement.

4. **Short test runs first** - run a few timesteps on CPU at low resolution.
   Check for NaNs, expected flow development, and meaningful output.

5. **Progressive validation** - run a short simulation, visualize, check physics
   (flow direction, velocity magnitude, mixing patterns).

6. **Compare to paper figures** - match figure format, colormaps, axis ranges,
   time snapshots. Compute the same diagnostics.

## Common Issues

- **NaN blowups**: timestep too large, unstable ICs, or `if`/`else` on GPU (use `ifelse`)
- **Nothing happening**: wrong sign on buoyancy anomaly, ICs not applied, forcing inactive
- **Wrong flow direction**: check coordinate conventions (upslope vs downslope)
- **GPU issues**: avoid branching, ensure type stability
