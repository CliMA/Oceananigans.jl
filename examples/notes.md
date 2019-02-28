# Convection into a linearly stratified fluid

We are performing 'large eddy simulations' (actually, direct numerical 
simulations with artificially elevated viscosity and diffusivity)
of convection into a stratified fluid. 
Our parameters regime corresponds roughly to conditions that
occur in the North Atlantic ocean in winter.

## Parameters:

* ν : 10^(-2)       (m^2 / s),   viscocity
* κ : 10^(-2)       (m^2 / s),   thermal diffusivity
* N2: 6.4x10^(-7)   (1/s^2),     stratification at the bottom boundary
* f : 10^(-4)       (1/s),       coriolis parameter
* Fb: 2x10^(-7)     (m^2 / s^3), bouyancy flux at the top
* Lx: 500           (m),         domain size in the x-direction
* Ly: 500           (m),         domain size in the y-direction
* Lz: 500           (m),         domain size in the z-direction
