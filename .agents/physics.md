# Physics Domain Knowledge

## Ocean and Fluid Dynamics

- Incompressible/Boussinesq approximation for ocean flows
- Hydrostatic approximation for large-scale flows
- Free surface dynamics with implicit/explicit time stepping
- Coriolis effects and planetary rotation
- Stratification and buoyancy-driven flows
- Turbulence modeling via LES and eddy viscosity closures

## Numerical Methods

- Finite volume on structured grids (Arakawa C-grid)
- Staggered grid locations: velocities at cell faces, tracers at cell centers
- Various advection schemes: centered, upwind, WENO
- Pressure Poisson solver for incompressibility constraint
- Time stepping: RungeKutta, Adams-Bashforth, Quasi-Adams-Bashforth
- Take care of staggered grid location when writing operators or designing diagnostics
