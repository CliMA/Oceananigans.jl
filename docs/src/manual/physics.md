# Navier-Stokes and tracer conservation equations

Oceananigans.jl solves the incompressible Navier-Stokes equations and an arbitrary 
number of tracer conservation equations. 
Physics associated with individual terms in the momentum and tracer conservation 
equations --- the background rotation rate of the equation's reference frame, 
gravitational effects associated with buoyant tracers under the Boussinesq 
approximation[^1], generalized stresses and tracer fluxes associated with viscous and 
diffusive physics, and arbitrary "forcing functions" --- are determined by the whims of the 
user.

[^1]: Named after Boussinesq (1903) although used earlier by Oberbeck (1879), the Boussinesq
      approximation neglects density differences in the momentum equation except when associated 
      with the gravitational term. It is an accurate approximation for many flows, and especially 
      so for oceanic flows where density differences are very small. See Vallis (2017, section 2.4) 
      for an oceanographic introduction to the Boussinesq equations and Vallis (2017, Section 2.A) 
      for an asymptotic derivation. See Kundu (2015, Section 4.9) for an engineering 
      introduction.

## Coordinate system and notation

Oceananigans.jl is formulated in a Cartesian coordinate system 
$\bm{x} = (x, y, z)$ with unit vectors $\bm{\hat x}$, $\bm{\hat y}$, and $\bm{\hat z}$, 
where $\bm{\hat x}$ points east, $\bm{\hat y}$ points north, and $\bm{\hat z}$ points 'upward', 
opposite the direction of gravitational acceleration. 
We denote time with $t$, partial derivatives with respect to time $t$ or a coordinate $x$ 
with $\partial_t$ or $\partial_x$, and denote the gradient operator 
$\bm{\nabla} \equiv \partial_x \bm{\hat x} + \partial_y \bm{\hat y} + \partial_z \bm{\hat z}$.
We use $u$, $v$, and $w$ to denote the east, north, and vertical velocity components, 
such that $\bm{u} = u \bm{\hat x} + v \bm{\hat y} + w \bm{\hat z}$.

## The Boussinesq approximation

In Oceananigans.jl the fluid density $\rho$ is, in general, decomposed into three 
components: 
```math
    \rho(\bm{x}, t) = \rho_0 + \rho_*(z) + \rho'(\bm{x}, t) \, ,
```
where $\rho_0$ is a constant 'reference' density, $\rho_*(z)$ is a background density 
profile typically associated with the hydrostatic compression of seawater in the deep ocean, 
and $\rho'(\bm{x}, t)$ is the dynamic component of density corresponding to inhomogeneous 
distributions of a buoyant tracer such as temperature or salinity.
The fluid *buoyancy*, associated with the buoyant acceleration of fluid, is 
defined in terms of $\rho'$ as
```math
    b = - \frac{g \rho'}{\rho_0} \, ,
```
where $g$ is gravitational acceleration.

The Boussinesq approximation is valid when $\rho_* + \rho' \ll \rho_0$, which implies the 
fluid is approximately *incompressible*[^2]
In this case, the mass conservation equation reduces to the continuity equation
```math
    \bm{\nabla} \bm{\cdot} \bm{u} = \partial_x u + \partial_y v + \partial_z w = 0 \, .
    \tag{eq:continuity}
```

[^2]: Incompressible fluids do not support acoustic waves.

## The momentum conservation equation

The equations governing the conservation of momentum in a rotating fluid, including buoyancy
via the Boussinesq approximation and including the averaged effects of surface gravity waves
at the top of the domain via the Craik-Leibovich approximation are
```math
    \partial_t \bm{u} + \left ( \bm{u} \bm{\cdot} \bm{\nabla} \right ) \bm{u} 
        + \left ( \bm{f} - \bm{\nabla} \times \bm{u}^S \right ) \times \bm{u} = - \bm{\nabla} \phi + b \bm{\hat z} 
        - \bm{\nabla} \bm{\cdot} \bm{\tau} - \partial_t \bm{u}^S + \bm{F_u} \, ,
    \tag{eq:momentum}
```
where $b$ is buoyancy, $\bm{\tau}$ is the kinematic stress tensor, $\bm{F_u}$
denotes an internal forcing of the velocity field $\bm{u}$, $\phi$ is the potential 
associated with kinematic and constant hydrostatic contributions to pressure, 
$\bm{u}^S$ is the 'Stokes drift' velocity field associated with surface gravity waves,
and $\bm{f}$ is *Coriolis parameter*, or the background vorticity associated with the 
specified rate of rotation of the frame of reference.

## The tracer conservation equation

The conservation law for tracers in Oceananigans.jl is
```math
    \partial_t c + \bm{u} \bm{\cdot} \bm{\nabla} c = - \bm{\nabla} \bm{\cdot} \bm{q}_c + F_c \, ,
    \tag{eq:tracer}
```
where $\bm{q}_c$ is the diffusive flux of $c$ and $F_c$ is an arbitrary source term.
Oceananigans.jl permits arbitrary tracers and thus an arbitrary number of tracer 
equations to be solved simultaneously with the momentum equations.

## Buoyancy model and equations of state

The buoyancy model determines the relationship between tracers and the buoyancy $b$ in the momentum equation.

### Buoyancy tracer

The simplest buoyancy model uses buoyancy $b$ itself as a tracer: $b$ obeys the tracer
conservation equation and is used directly in the momentum equations in the momentum equation.

### Seawater buoyancy

For seawater buoyancy is, in general, modeled as a function of conservative temperature 
$\theta$, absolute salinity $S$, and depth below the ocean surface $d$ via
```math
    b = - \frac{g}{\rho_0} \rho' \left (\theta, S, d \right ) \, ,
    \tag{eq:seawater-buoyancy}
```
where $g$ is gravitational acceleration, $\rho_0$ is the reference density.
The function $\rho'(\theta, S, d)$ in the seawater buoyancy relationship that links conservative temperature, 
salinity, and depth to the density perturbation is called the *equation of state*.
Both $\theta$ and $S$ obey the tracer conservation equation.

#### Linear equation of state

Buoyancy is determined from a linear equation of state via
```math
    b = g \left ( \alpha_\theta \theta - \beta_S S \right ) \, ,
```
where $g$ is gravitational acceleration, $\alpha_\theta$ is the thermal expansion coefficient, 
and $\beta_S$ is the haline contraction coefficient.

#### Nonlinear equation of state 

Buoyancy is determined by the simplified equations of state introduced by Roquet et al (2015).

## Coriolis forces

The Coriolis model controls the manifestation of the term $\bm{f} \times \bm{u}$ in the momentum equation.

### The "$f$-plane" approximation

Under an $f$-plane approximation[^3] the reference frame in which 
the momentum and tracer equations are are solved rotates at a constant rate around a 
vertical axis, such that 
```math
    \bm{f} = f \bm{\hat z}
```
where $f$ is constant and determined by the user. 

[^3]: The $f$-plane approximation is used to model the effects of Earth's rotation on 
      anisotropic fluid motion in a plane tangent to the Earth's surface. In this case, $\bm{f}$ is 
      ```math
          \bm{f} \approx \frac{4 \pi}{\text{day}} \sin \varphi \bm{\hat z} \, , $
      ```
      where $\phi$ is latitude and the Earth's rotation rate is approximately $2 \pi / \text{day}$.
      This approximation neglects the vertical component of Earth's rotation vector at $\varphi$.

### The $\beta$-plane approximation

Under the $\beta$-plane approximation, the rotation axis is vertical as for the 
$f$-plane approximation, but $f$ is expanded in a Taylor series around a central latitude such that 
```math
    \bm{f} = \left ( f_0 + \beta y \right ) \bm{\hat z} \, ,
```
where $f_0$ is the planetary vorticity at some central latitude, and $\beta$ is the 
planetary vorticity gradient.
The $\beta$-plane model is not periodic in $y$ and thus can be used only in domains that 
are bounded in the $y$-direction.

## Turbulence closures

The turbulence closure selected by the user determines the form of stress divergence 
$\bm{\nabla} \bm{\cdot} \bm{\tau}$ and diffusive flux divergence 
$\bm{\nabla} \bm{\cdot} \bm{q}_c$ in the momentum and tracer conservation equations.

### Constant isotropic diffusivity

In a constant isotropic diffusivity model, the kinematic stress tensor is defined
```math
\tau_{ij} = - \nu \Sigma_{ij} \, ,
```
where $\nu$ is a constant viscosity and 
$\Sigma_{ij} \equiv \tfrac{1}{2} \left ( u_{i, j} + u_{j, i} \right )$ is the strain-rate 
tensor. The divergence of $\bm{\tau}$ is then
```math
\bm{\nabla} \bm{\cdot} \bm{\tau} = -\nu \bm{\nabla}^2 \bm{u} \, .
```
Similarly, the diffusive tracer flux is $\bm{q}_c = - \kappa \bm{\nabla} c$ for tracer 
diffusivity $\kappa$, and the diffusive tracer flux divergence is
```math
\bm{\nabla} \bm{\cdot} \bm{q}_c = - \kappa \bm{\nabla}^2 c \, .
```
Each tracer may have a unique diffusivity $\kappa$.

### Constant anisotropic diffusivity

In Oceananigans.jl, a constant anisotropic diffusivity implies a constant tensor 
diffusivity $\nu_{j k}$ and stress $\bm{\tau}_{ij} = \nu_{j k} u_{i, k}$ with non-zero 
components $\nu_{11} = \nu_{22} = \nu_h$ and $\nu_{33} = \nu_v$.
With this form the kinematic stress divergence becomes
```math
\bm{\nabla} \bm{\cdot} \bm{\tau} = - \left [ \nu_h \left ( \partial_x^2 + \partial_y^2 \right ) 
                                    + \nu_v \partial_z^2 \right ] \bm{u} \, ,
```
and diffusive flux divergence
```math
\bm{\nabla} \bm{\cdot} \bm{q}_c = - \left [ \kappa_{h} \left ( \partial_x^2 + \partial_y^2 \right ) 
                                    + \kappa_{v} \partial_z^2 \right ] c \, .
```
in terms of the horizontal viscosities and diffusivities $\nu_h$ and $\kappa_{h}$ and the 
vertical viscosity and diffusivities $\nu_v$ and $\kappa_{v}$.
Each tracer may have a unique diffusivity components $\kappa_h$ and $\kappa_v$.

### Constant anisotropic biharmonic diffusivity

In Oceananigans.jl, a constant anisotropic biharmonic diffusivity implies a constant tensor 
diffusivity $\nu_{j k}$ and stress $\bm{\tau}_{ij} = \nu_{j k} \partial_k^3 u_i$ with non-zero 
components $\nu_{11} = \nu_{22} = \nu_h$ and $\nu_{33} = \nu_v$.
With this form the kinematic stress divergence becomes
```math
\bm{\nabla} \bm{\cdot} \bm{\tau} = - \left [ \nu_h \left ( \partial_x^2 + \partial_y^2 \right )^2 
                                    + \nu_v \partial_z^4 \right ] \bm{u} \, ,
```
and diffusive flux divergence
```math
\bm{\nabla} \bm{\cdot} \bm{q}_c = - \left [ \kappa_{h} \left ( \partial_x^2 + \partial_y^2 \right )^2 
                                    + \kappa_{v} \partial_z^4 \right ] c \, .
```
in terms of the horizontal biharmonic viscosities and diffusivities $\nu_h$ and $\kappa_{h}$ and the 
vertical biharmonic viscosity and diffusivities $\nu_v$ and $\kappa_{v}$.
Each tracer may have a unique diffusivity components $\kappa_h$ and $\kappa_v$.

### Smagorinsky-Lilly turbulence closure

In the turbulence closure proposed by Lilly (1962) and Smagorinsky (1963), 
the subgrid stress associated with unresolved turbulent motions is modeled diffusively via
```math
\tau_{ij} = \nu_e \Sigma_{ij} \, ,
```
where $\Sigma_{ij} = \tfrac{1}{2} \left ( u_{i, j} + u_{j, i} \right )$ is the resolved 
strain rate. 
The eddy viscosity is given by
```math
    \nu_e = \left ( C \Delta_f \right )^2 \sqrt{ \Sigma^2 } \, \Upsilon(Ri) + \nu \, ,
    \tag{eq:smagorinsky-viscosity}
```
where $\Delta_f$ is the "filter width" associated with the finite volume grid spacing, 
$C$ is a user-specified model constant, $\Sigma^2 \equiv \Sigma_{ij} \Sigma{ij}$, and 
$\nu$ is a constant isotropic background viscosity.
The factor $\Upsilon(Ri)$ reduces $\nu_e$ in regions of 
strong stratification where the resolved gradient Richardson number 
$Ri \equiv N^2 / \Sigma^2$ is large via
```math
    \Upsilon(Ri) = \sqrt{1 - \min \left ( 1, C_b N^2 / \Sigma^2 \right )} \, ,
```
where $N^2 = \max \left (0, \partial_z b \right )$ is the squared buoyancy frequency for stable
stratification with $\partial_z b > 0$ and $C_b$ is a user-specified constant.
Roughly speaking, the filter width for the Smagorinsky-Lilly closure is taken as
```math
\Delta_f(\bm{x}) = \left ( \Delta x \Delta y \Delta z \right)^{1/3} \, ,
```
where $\Delta x$, $\Delta y$, and $\Delta z$ are the grid spacing in the 
$\bm{\hat x}$, $\bm{\hat y}$, and $\bm{\hat z}$ directions at location $\bm{x} = (x, y, z)$.

The effect of subgrid turbulence on tracer mixing is also modeled diffusively via
```math
\bm{q}_c = \kappa_e \bm{\nabla} c \, ,
```
where the eddy diffusivity $\kappa_e$ is
```math
\kappa_e = \frac{\nu_e - \nu}{Pr} + \kappa \, ,
```
where $Pr$ is a turbulent Prandtl number and $\kappa$ is a constant isotropic background diffusivity.
Both $Pr$ and $\kappa$ may be set independently for each tracer.

### Anisotropic minimum dissipation (AMD) turbulence closure

Oceananigans.jl uses the anisotropic minimum dissipation (AMD) model proposed by 
Verstappen18 and described and tested by Vreugdenhil18. 
The AMD model uses an eddy diffusivity hypothesis similar the Smagorinsky-Lilly model.
In the AMD model, the eddy viscosity and diffusivity for each tracer are defined in terms 
of eddy viscosity and diffusivity \emph{predictors}
$\nu_e^\dagger$ and $\kappa_e^\dagger$, such that
```math
    \nu_e = \max \left ( 0, \nu_e^\dagger \right ) + \nu
    \quad \text{and} \quad
    \kappa_e = \max \left ( 0, \kappa_e^\dagger \right ) + \kappa
```
to ensure that $\nu_e \ge 0$ and $\kappa_e \ge 0$, where $\nu$ and $\kappa$ are the 
constant isotropic background viscosity and diffusivities for each tracer.
The eddy viscosity predictor is
```math
    \tag{eq:nu-dagger}
    \nu_e^\dagger = -(C \Delta_f)^2
    \frac
        {\left( \hat{\partial}_k \hat{u}_i \right) \left( \hat{\partial}_k \hat{u}_j \right) \hat{\Sigma}_{ij}
        + C_b \hat{\delta}_{i3} \left( \hat{\partial}_k \hat{u_i} \right) \hat{\partial}_k b}
        {\left( \hat{\partial}_l \hat{u}_m \right) \left( \hat{\partial}_l \hat{u}_m \right)}
```
while the eddy diffusivity predictor for tracer $c$ is
```math
    \tag{eq:kappa-dagger}
    \kappa_e^\dagger = -(C \Delta_f)^2
    \frac
        {\left( \hat{\partial}_k \hat{u}_i \right) \left( \hat{\partial}_k c \right) \hat{\partial}_i c}
        {\left( \hat{\partial}_l c \right) \left( \hat{\partial}_l c \right)} \, .
```
In the definitions of the eddy viscosity and eddy diffusivity predictor, $C$ and $C_b$ are 
user-specified model constants, $\Delta_f$ is a "filter width" associated with the finite volume 
grid spacing, and the hat decorators on partial derivatives, velocities, and the Kronecker 
delta $\hat \delta_{i3}$ are defined such that
```math
    \hat \partial_i \equiv \Delta_i \partial_i, \qquad
    \hat{u}_i(x, t) \equiv \frac{u_i(x, t)}{\Delta_i}, \quad \text{and} \quad
    \hat{\delta}_{i3} \equiv \frac{\delta_{i3}}{\Delta_3} \, .
```
A velocity gradient, for example, is therefore 
$\hat{\partial}_i \hat{u}_j(x, t) = \frac{\Delta_i}{\Delta_j} \partial_i u_j(x, t)$, 
while the normalized strain tensor is
```math
    \hat{\Sigma}_{ij} =
        \frac{1}{2} \left[ \hat{\partial}_i \hat{u}_j(x, t) + \hat{\partial}_j \hat{u}_i(x, t) \right] \, .
``` 
The filter width $\Delta_f$ in that appears in the viscosity and diffusivity predictors
is taken as the square root of the harmonic mean of the squares of the filter widths in 
each direction:
```math
    \frac{1}{\Delta_f^2} = \frac{1}{3} \left(   \frac{1}{\Delta x^2} 
                                              + \frac{1}{\Delta y^2} 
                                              + \frac{1}{\Delta z^2} \right) \, .
```
The constant $C_b$ permits the "buoyancy modification" term it multiplies to be omitted 
from a calculation.
By default we use the model constants $C=1/12$ and $C_b=0$.

## Surface gravity waves and the Craik-Leibovich approximation

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
therefore, are not present in the resolved velocity field $\bm{u}$, and only the steady, 
averaged effect of surface waves that manifests over several or more wave oscillations are modeled.

In Oceananigans.jl with surface waves, the resolved velocity field $\bm{u}$ is the Lagrangian-mean 
velocity field.
The Lagrangian-mean velocity field at a particular location $(x, y, z)$ is average velocity of a 
fluid particle whose average position is $(x, y, z)$ at time $t$.
The average position of a fluid particle $\bm{\xi}(t) = (\xi, \eta, \zeta)$ is thus governed by
```math
    \partial_t \bm{\xi} + \bm{u}(\bm{\xi}, t) \bm{\cdot} \bm{\nabla} \bm{\xi} = \bm{u}(\bm{\xi}, t) \, ,
```
which is the same relationship that holds when surface waves are not present and $\bm{u}$ ceases
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

The Lagrangian-mean velocity field $\bm{u}$ contrasts with the Eulerian-mean velocity field $\bm{u}^E$, 
which is the fluid velocity averaged at the fixed Eulerian position $(x, y, z)$.
The surface wave Stokes drift field supplied by the user is, in fact, defined
by the difference between the Eulerian- and Lagrangian-mean velocity:
```math
    \bm{u}^S \equiv \bm{u} - \bm{u}^E \, .
```
The Stokes drift velocity field is typically prescribed for idealized scenarios, or determined
from a wave model for the evolution of surface waves under time-dependent atmospheric winds
in more realistic cases.

## Boundary conditions

In Oceananigans.jl the user may impose \textit{no-penetration}, \textit{flux}, 
\textit{gradient} (Neumann), and \textit{value} (Dirichlet) boundary conditions in bounded, 
non-periodic directions.
Note that the only boundary condition available for a velocity field normal to the bounded 
direction is \textit{no-penetration}.

### Flux boundary conditions

A flux boundary condition prescribes flux of a quantity normal to the boundary. 
For a tracer $c$ this corresponds to prescribing 
```math
q_c \, |_b \equiv \bm{q}_c \bm{\cdot} \hat{\bm{n}} \, |_{\partial \Omega_b} \, , 
```
where $\partial \Omega_b$ is an external boundary.

### Gradient (Neumann) boundary condition 

A gradient boundary condition prescribes the gradient of a field normal to the boundary. 
For a tracer $c$ this prescribes 
```math
\gamma \equiv \bm{\nabla} c \bm{\cdot} \hat{\bm{n}} \, |_{\partial \Omega_b} \, .
```

### Value (Dirichlet) boundary condition 

A value boundary condition prescribes the value of a field on a boundary; for a tracer this 
prescribes 
```math
c_b \equiv c \, |_{\partial \Omega_b} \, .
```

### No penetration boundary condition 

A no penetration boundary condition prescribes the velocity component normal to a boundary to be 0,
so that
```math
\bm{\hat{n}} \bm{\cdot} \bm{u} \, |_{\partial \Omega_b} = 0 \, .
```

