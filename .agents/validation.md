# Implementing Validation Cases / Reproducing Paper Results

When implementing a simulation from a published paper:

## 1. Parameter Extraction

- **Read the paper carefully** and extract ALL parameters: domain size, resolution, physical constants,
  boundary conditions, initial conditions, forcing, closure parameters
- Look for parameter tables (often "Table 1" or similar)
- Check figure captions for additional details
- Note the coordinate system and conventions used

## 2. Geometry Verification (BEFORE running long simulations)

- **Always visualize the grid/domain geometry first**
- Check that:
  - Domain extents match the paper
  - Topography/immersed boundaries are correct
  - Coordinate orientations match (which direction is "downslope"?)
- Compare your geometry plot to figures in the paper

## 3. Initial Condition Verification

- After setting initial conditions, check:
  - `minimum(field)` and `maximum(field)` make physical sense
  - Spatial distribution looks correct (visualize if needed)
  - Dense water is where it should be, stratification is correct, etc.

## 4. Short Test Runs

Before running a long simulation:
- Run for a few timesteps on CPU at low resolution
- Verify:
  - No NaNs appear (check `maximum(abs, u)` etc.)
  - Flow is developing as expected (velocities increasing from zero)
  - Output files contain meaningful data
- Then test on GPU to catch GPU-specific issues

## 5. Progressive Validation

- Run a short simulation (e.g., 1 hour sim time) and visualize
- Check that the physics looks right:
  - Dense water flowing in the correct direction?
  - Velocities reasonable magnitude?
  - Mixing/entrainment happening where expected?
- Compare to early-time figures in the paper if available

## 6. Comparison to Paper Figures

- Create visualizations that match the paper's figure format
- Use the same colormaps, axis ranges, and time snapshots if possible
- Quantitative comparison: compute the same diagnostics as the paper

## 7. Common Issues

- **NaN blowups**: Usually from timestep too large, unstable initial conditions,
  or if-else statements on GPU (use `ifelse` instead)
- **Nothing happening**: Check that buoyancy anomaly has the right sign,
  that initial conditions are actually applied, that forcing is active
- **Wrong direction of flow**: Check coordinate conventions (is y increasing
  upslope or downslope?)
- **GPU issues**: Avoid branching, ensure type stability, use `randn()` carefully
