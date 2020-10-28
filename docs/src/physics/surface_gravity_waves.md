# Surface gravity waves and the Craik-Leibovich approximation

In Oceananiagns.jl, users model the effects of surface waves by specifying spatial and
temporal gradients of the Stokes drift velocity field.
At the moment, only uniform unidirectional Stokes drift fields are supported, in which case
```math
    \bm{u}^S = u^S(z, t) \hat{\bm{x}} + v^S(z, t) \hat{\bm{y}} \, .
```
Surface waves are modeled in Oceananigans.jl by the Craik-Leibovich approximation,
which governs interior motions under a surface gravity wave field that have been time- or
phase-averaged over the rapid oscillations of the surface waves.
The oscillatory vertical and horizontal motions associated with surface waves themselves,
therefore, are not present in the resolved velocity field ``\bm{u}``, and only the steady,
averaged effect of surface waves that manifests over several or more wave oscillations are modeled.

In Oceananigans.jl with surface waves, the resolved velocity field ``\bm{u}`` is the Lagrangian-mean
velocity field.
The Lagrangian-mean velocity field at a particular location ``(x, y, z)`` is average velocity of a
fluid particle whose average position is ``(x, y, z)`` at time ``t``.
The average position of a fluid particle ``\bm{\xi}(t) = (\xi, \eta, \zeta)`` is thus governed by
```math
    \partial_t \bm{\xi} + \bm{u}(\bm{\xi}, t) \bm{\cdot} \bm{\nabla} \bm{\xi} = \bm{u}(\bm{\xi}, t) \, ,
```
which is the same relationship that holds when surface waves are not present and ``\bm{u}`` ceases
to be an averaged velocity field.
The simplicity of the governing equations for Lagrangian-mean momentum is the main reason we
use a Lagrangian-mean formulation in Oceananigans.jl, rather than an Eulerian-mean formulation:
for example, the tracer conservation equation is unchanged by the inclusion of surface wave effects.
Moreover, because the effect of surface waves manifests either as a bulk forcing of
Lagrangian-mean momentum or as a modification to the effective background rotation rate of
the interior fluid similar to any bulk forcing or Coriolis force, we do not explicitly include the
effects of surface waves in turbulence closures that model the effects of subgrid turbulence.
More specifically, the effect of steady surface waves does not effect the conservation of
Lagrangian-mean turbulent kinetic energy.

The Lagrangian-mean velocity field ``\bm{u}`` contrasts with the Eulerian-mean velocity field ``\bm{u}^E``,
which is the fluid velocity averaged at the fixed Eulerian position ``(x, y, z)``.
The surface wave Stokes drift field supplied by the user is, in fact, defined
by the difference between the Eulerian- and Lagrangian-mean velocity:
```math
    \bm{u}^S \equiv \bm{u} - \bm{u}^E \, .
```
The Stokes drift velocity field is typically prescribed for idealized scenarios, or determined
from a wave model for the evolution of surface waves under time-dependent atmospheric winds
in more realistic cases.
