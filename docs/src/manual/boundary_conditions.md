# Boundary conditions

In finite volume methods, boundary conditions are imposed weakly by adding viscous or diffusive fluxes into grid cells
adjacent to the boundary which have the effect of imposing the desired boundary condition.

Without loss of generality we will consider how to impose boundary conditions in the $y$-direction assuming it is
wall-bounded. The viscous or diffusive fluxes for a variable $c$ show up in the evolution equation as
```math
    \partial_t c = - \partial_y (\Gamma \partial_y c)
```
where the flux is given by $\Gamma \partial_y c$ and $\Gamma$ may represent the kinematic viscosity if $c$ is a
velocity, or a diffusivity if $c$ is a tracer.

Upon discretizing the right-hand-side of \eqref{eq:flux-tendency} at grid cell $j$ we get
```math
    \partial_y (\Gamma \partial_y c)_j = \frac{\Gamma_j (\partial_y c)_j - \Gamma_{j-1} (\partial_y c)_{j-1}}{\Delta y}
```
but by default, we assume there are no fluxes at the walls so that at the first grid cell we get
```math
    \partial_y (\Gamma \partial_y c)_1 = \frac{\Gamma_1 (\partial_y c)_1}{\Delta y} - f_w
```
where $f_w$ is the flux at the wall, assumed to be $f_w = 0$ by default.

## Flux boundary conditions

Suppose we would like to impose a flux at the wall, then this is simply done by setting $f_w$ to be non-zero.

## Neumann boundary conditions

Suppose we want to set the gradient $\partial c/\partial y$ at the wall, which we will denote ``G = (\partial_y c)_w``.
Then in effect we would like to impose ``\Gamma_w (\partial_y c)_w``. This can be done by setting
$f_w = \Gamma G / \Delta y$. An unfortunate consequence of this is that $\Gamma_w$ is undefined and left as a modeling
choice. This is not a problem unless $\Gamma$ varies, such as when using an LES closure.

## Dirchlet boundary conditions
Suppose we want to set $c$ to be some value $V$ at the wall. Then this can be accomplished by discretizing
```math
    (\partial_y c)_w = \frac{c_1 - V}{\Delta y / 2} \quad \implies \quad f_w = \frac{2\Gamma_w}{\Delta y^2} (V - c_1)
```
where we divide by $\Delta y / 2$ because that is the distance between the center of the first grid cell and the wall.
This can be interpreted as adding a relaxation term to the evolution equation with a very short relaxation timescale
of $\Delta y^2 / \Gamma_w$. Again, $\Gamma_w$ is undefined at the wall.
