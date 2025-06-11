# Generalized vertical coordinate

For `HydrostaticFreeSurfaceModel()`, the user can choose between a `ZCoordinate` and a `ZStar` vertical coordinate.
A `ZStar` vertical coordinate conserves tracers and volume with the grid following the evolution of the free surface in the domain [adcroft2004rescaled](@citet).
To obtain the (discrete) equations evolved  in a general framework where the vertical coordinate is moving, we perform a scaling of the continuous primitive equations to a generalized coordinate ``r(x, y, z, t)``.

We have that:

```math
\begin{alignat}{2}
& \frac{\partial \phi}{\partial s}\bigg\rvert_{z} && =  \frac{\partial \phi}{\partial s}\bigg\rvert_{r} + \frac{\partial \phi}{\partial r} \frac{\partial r}{\partial s} \\ 
& \frac{\partial \phi}{\partial z} && = \frac{1}{\sigma}\frac{\partial \phi}{\partial r}
\end{alignat}
```
where $s = x, y, t$ and 
```math
\begin{equation}
\sigma = \frac{\partial z}{\partial r} \bigg\rvert_{x, y, t}
\end{equation}
```
We can also write the spatial derivatives of the ``r``-coordinate as follows
```math
\frac{\partial r}{\partial x}\bigg\rvert_{y, z, t} = - \frac{\partial z}{\partial x}\bigg\rvert_{y, s, t} \frac{1}{\sigma}
```
Such that the chain rule above for horizontal spatial derivatives (``x`` and ``y``) becomes

```math
\begin{alignat}{2}
& \frac{\partial \phi}{\partial x}\bigg\rvert_{z} && = \frac{\partial \phi}{\partial x}\bigg\rvert_{r} - \frac{1}{\sigma}\frac{\partial \phi}{\partial r} \frac{\partial z}{\partial x}  \\ 
& \frac{\partial \phi}{\partial y}\bigg\rvert_{z} && = \frac{\partial \phi}{\partial y}\bigg\rvert_{r} - \frac{1}{\sigma}\frac{\partial \phi}{\partial r} \frac{\partial z}{\partial y}  
\end{alignat}
```
## Continuity Equation
Following the above ruleset, the divergence of the velocity field can be rewritten as
```math
\begin{align}
\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{u} & = \frac{\partial u}{\partial x} \bigg\rvert_{z} + \frac{\partial v}{\partial y} \bigg\rvert_{z} + \frac{\partial w}{\partial z} \\
& = \frac{\partial u}{\partial x} \bigg\rvert_{r} + \frac{\partial v}{\partial y} \bigg\rvert_{r} - \frac{1}{\sigma} \left( \frac{\partial u}{\partial r} \frac{\partial z}{\partial x} + \frac{\partial v}{\partial r} \frac{\partial z}{\partial y}  - \frac{\partial w}{\partial r} \right) \\ 
& = \frac{1}{\sigma} \left( \frac{\partial \sigma u}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial y} \bigg\rvert_{r} - u \frac{\partial \sigma}{\partial x} \bigg\rvert_{r} -  v \frac{\partial \sigma}{\partial y} \bigg\rvert_{r} \right)- \frac{1}{\sigma} \left( \frac{\partial u}{\partial r} \frac{\partial z}{\partial x} + \frac{\partial v}{\partial y} \frac{\partial z}{\partial y}  - \frac{\partial w}{\partial r} \right)
\end{align}
```
We can rewrite $\partial_x \sigma \rvert_r = \partial_r(\partial_x z)$ and the same for the ``y`` direction. Then the above yields
```math
\begin{align}
\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{u} & = \frac{1}{\sigma} \left( \frac{\partial \sigma u}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial y}\bigg\rvert_{r} \right)- \frac{1}{\sigma} \left( u \frac{\partial^2 z}{\partial x \partial r} +  v \frac{\partial^2 z}{\partial y \partial r} + \frac{\partial u}{\partial r} \frac{\partial z}{\partial x} + \frac{\partial v}{\partial y} \frac{\partial z}{\partial y}  - \frac{\partial w}{\partial r} \right) \\
& = \frac{1}{\sigma} \left( \frac{\partial \sigma u}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial y}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial}{\partial r} \left( u \frac{\partial z}{\partial x} +  v \frac{\partial z}{\partial y} + w \right) 
\end{align}
```
Here, $w$ is the vertical velocity referenced to the ``z`` coordinate. We can define the vertical velocity $w_r$ of the ``r`` surface referenced to the ``z`` coordinate as
```math
w_r = \frac{\partial z}{\partial t} \bigg\rvert_r + u \frac{\partial z}{\partial x} +  v \frac{\partial z}{\partial y}
```
Then, the vertical velocity across the ``r`` surfaces is the difference of $w$ and $w_r$
```math
\omega = w - w_r = w - \frac{\partial z}{\partial t} \bigg\rvert_r - u \frac{\partial z}{\partial x} - v \frac{\partial z}{\partial y}
```
Therefore, adding the definition of $\omega$ to the velocity divergence we get
```math
\begin{align}
\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{u} & = \frac{1}{\sigma} \left( \frac{\partial \sigma u}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial y}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial}{\partial r} \left( \omega + \frac{\partial z}{\partial t}\bigg\rvert_r \right) \\
& = \frac{1}{\sigma} \left( \frac{\partial \sigma u}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial y}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial \omega}{\partial r} + \frac{1}{\sigma} \frac{\partial \sigma}{\partial t}
\end{align}
```
which finally leads to the continuity equation
```math
\frac{\partial \sigma}{\partial t} + \frac{\partial \sigma u}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial y}\bigg\rvert_{r}  + \frac{\partial \omega}{\partial r} = 0
```
### Finite volume discretization of the continuity equation

It is usefull to think about this equation in the discrete form in a finite volume staggered C-grid framework, where we integrate over a volume $V_r = \Delta x \Delta y \Delta r$ remembering that in the discrete $\Delta z = \sigma \Delta r$. The indices `i`, `j`, `k` correspond to the `x`, `y`, and the vertical direction.
```math
\frac{1}{V_r}\int_{V_r} \frac{\partial \sigma}{\partial t} \, \mathrm{d}V + \frac{1}{V_r} \int_{V_r} \left(\frac{\partial \sigma u}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial y}\bigg\rvert_{r}  + \frac{\partial \omega}{\partial r}\right) \, \mathrm{d}V = 0
```
Using the divergence theorem, and introducing the notation of cell-average values $V_r^{-1} \int_{V_r} \phi \,\mathrm{d}V = \overline{\phi}$
```math
\frac{\partial \overline{\sigma}}{\partial t} + \frac{1}{\Delta x\Delta y \Delta r} \left( \Delta y \Delta r \sigma u\rvert_{i-1/2}^{i+1/2} + \Delta x \Delta r \sigma v\rvert_{j-1/2}^{j+1/2} \right ) + \frac{\overline{\omega}_{k+1/2} - \overline{\omega}_{k-1/2}}{\Delta r} = 0
```
The above equation is used to diagnose the vertical velocity (in `r` space) given the grid velocity and the horizontal velocity divergence:
```math
\overline{\omega}_{k+1/2} = \overline{\omega}_{k-1/2} + \Delta r \frac{\partial \overline{\sigma}}{\partial t} + \frac{1}{Az} \left( \mathcal{U}\rvert_{i-1/2}^{i+1/2} + \mathcal{V}\rvert_{j-1/2}^{j+1/2} \right )
```
where $\mathcal{U} = Axu$, $\mathcal{V} = Ayv$, $Ax = \Delta y \Delta z$, $Ay = \Delta x \Delta z$, and $Az = \Delta x \Delta y$.

## Tracer equations
The tracer equation with vertical diffusion reads
```math
\frac{\partial T}{\partial t}\bigg\rvert_{z} + \boldsymbol{\nabla} \cdot \boldsymbol{u}T = \frac{\partial}{\partial z} \left( \kappa \frac{\partial T}{\partial z} \right)
```
Using the same procedure we followed for the continuity equation, $\partial_t T\rvert_{z} + \boldsymbol{\nabla} \cdot \boldsymbol{u}T$ yields
```math
\begin{align}
\frac{\partial T}{\partial t}\bigg\rvert_{z} + \boldsymbol{\nabla} \cdot \boldsymbol{u}T & = \frac{\partial T}{\partial t}\bigg\rvert_{z} + \frac{1}{\sigma} \left( \frac{\partial \sigma u T}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v T}{\partial y}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial}{\partial r}\left( T\omega + T \frac{\partial z}{\partial t}\bigg\rvert_r \right)  \\

& = \frac{\partial T}{\partial t}\bigg\rvert_{z} + \frac{1}{\sigma} \left( \frac{\partial \sigma u T}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v T}{\partial y}\bigg\rvert_{r} \right) + \frac{1}{\sigma} T\left( \frac{\partial \omega}{\partial r} + \frac{\partial \sigma}{\partial t}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \left( \omega + \frac{\partial z}{\partial t}\bigg\rvert_r \right)\frac{\partial T}{\partial r}\\

& = \frac{\partial T}{\partial t}\bigg\rvert_{z} + \frac{1}{\sigma} \left( \frac{\partial \sigma u T}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v T}{\partial y}\bigg\rvert_{r} \right) + 
\frac{1}{\sigma} \frac{\partial \omega T}{\partial r} + \frac{1}{\sigma}T \frac{\partial \sigma}{\partial t}\bigg\rvert_{r} + \frac{1}{\sigma}  \frac{\partial z}{\partial t}\bigg\rvert_r \frac{\partial T}{\partial r} \\
\end{align}
```
We recover the time derivative of the tracer at constant `r` by rewriting the last term using the chain rule for a time derivative
```math
\frac{1}{\sigma}  \frac{\partial z}{\partial t}\bigg\rvert_r \frac{\partial T}{\partial r} = \frac{\partial r}{\partial t} \frac{\partial T}{\partial r} = \frac{\partial T}{\partial t}\bigg\rvert_{r} - \frac{\partial T}{\partial t}\bigg\rvert_{z}
```
As such, $\partial_t T\rvert_{z} + \boldsymbol{\nabla} \cdot \boldsymbol{u}T$ can be rewritten in ``r``-coordinates as
```math
\frac{\partial T}{\partial t}\bigg\rvert_{z} + \boldsymbol{\nabla} \cdot \boldsymbol{u}T = \frac{1}{\sigma}\frac{\partial \sigma T}{\partial t}\bigg\rvert_r + \frac{1}{\sigma} \left( \frac{\partial \sigma u T}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v T}{\partial y}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial \omega T}{\partial r}
```
We add vertical diffusion to the RHS to recover the tracer equation
```math
\frac{1}{\sigma}\frac{\partial \sigma T}{\partial t}\bigg\rvert_r + \frac{1}{\sigma} \left( \frac{\partial \sigma u T}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v T}{\partial y}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial T \omega}{\partial r} = \frac{1}{\sigma}\frac{\partial}{\partial r} \left( \kappa \frac{\partial T}{\partial z} \right)
```
### Finite-volume discretization of the tracer equation

We discretize the equation in a finite volume framework
```math
\frac{1}{V_r}\int_{V_r} \frac{1}{\sigma}\frac{\partial \sigma T}{\partial t} + \frac{1}{V_r} \int_{V_r} \left[ \frac{1}{\sigma} \left( \frac{\partial \sigma u T}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v T}{\partial y}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial \omega T}{\partial r}\right] \, \mathrm{d}V = \frac{1}{V_r}\int_{V_r} \frac{1}{\sigma}\frac{\partial}{\partial r} \left( \kappa \frac{\partial T}{\partial z} \right) \, \mathrm{d}V
```
leading to
```math
\frac{1}{\sigma}\frac{\partial \sigma \overline{T}}{\partial t} + \frac{\mathcal{U}T\rvert_{i-1/2}^{i+1/2} + \mathcal{V}T\rvert_{j-1/2}^{j+1/2} + \mathcal{W} T\rvert_{k-1/2}^{k+1/2}}{V} = \frac{1}{V} \left(\mathcal{K} \frac{\partial T}{\partial z}\bigg\rvert_{k-1/2}^{k+1/2} \right)
```
where $V = \sigma V_r = \Delta x \Delta y \Delta z$, $\mathcal{U} = Axu$, $\mathcal{V} = Ay v$, $\mathcal{W} = Az \omega$, and $\mathcal{K} = Az \kappa$.

In case of an explicit formulation of the diffusive fluxes, the time discretization of the above via Forward Euler yields
```math
\begin{equation}
T^{n+1} = \frac{\sigma^n}{\sigma^{n+1}}\left(T^n + \Delta t \, G^n \right) 
\end{equation}
```
where $G^n$ is tendency computed on the `z`-grid.

Note that in case of a multi-step method, like second-order Adams Bashorth, the grid at different time-steps must be accounted for, and the time discretization becomes
```math
\begin{equation}
T^{n+1} = \frac{1}{\sigma^{n+1}}\left[\sigma^n T^n + \Delta t \left(\frac{3}{2}\sigma^n G^n - \frac{1}{2} \sigma^{n-1} G^{n-1} \right)\right]
\end{equation}
```
For this reason, we store tendencies pre-multipled by $\sigma$ at their current time-level.
In case of an implicit discretization of the diffusive fluxes we first compute $T^{n+1}$ as in the above equation (where $G^n$ does not contain the diffusive fluxes).
Then the implicit step is done on a `z`-grid as if the grid was static, using the grid at $n+1$ which includes $\sigma^{n+1}$.

## Momentum equations in vector invariant form

The momentum equations solved in Primitive equations models read
```math
\frac{D \boldsymbol{u}_h}{Dt} \bigg\rvert_z + f\boldsymbol{z} \times \boldsymbol{u}_h = - \nabla p \rvert_z - g\nabla \eta \rvert_z + \frac{\partial }{\partial z} \left( \nu \frac{\partial \boldsymbol{u}_h}{\partial z}\right)
```
complemented by the hydrostatic relation
```math
\frac{\partial p}{\partial z} = b
```
Of the above, the Coriolis term is independent of the vertical frame of reference and the viscous stress is treated similarly to the diffusion of a tracer. In this derivation we focus on (1) the hydrostatic relation, (2) the material derivative in the momentum equation, and (3) the horizontal pressure gradient terms.

### Hydrostatic relation
This equation is simple to transform by using the definition of a `z`-derivative in `r`-coordinates
```math
\frac{\partial p}{\partial r} = \sigma b
```

### Material derivative in vector invariant form

We set out to transform in ``r``-coordinates the material derivative of the horizontal velocity in vector invariant form 
```math
\frac{D \boldsymbol{u}_h}{Dt} \bigg\rvert_z = \frac{\partial \boldsymbol{u}_h}{\partial t} \bigg\rvert_z + \zeta \boldsymbol{z} \times \boldsymbol{u}_h + \boldsymbol{\nabla}_h K + w \frac{\partial \boldsymbol{u}_h}{\partial z}
```
where $\boldsymbol{u}_h = (u, v)$ is the horizontal velocity, $\zeta = \partial_xv - \partial_y u$ is the vertical vorticity and $K = (u^2 + v^2)/2$ is the horizontal kinetic energy.
In particular we will focus on the $u$ component of the velocity. The derivation of the $v$ component follows the same steps. In particular, we are transforming
```math
\begin{align}
\frac{D u}{Dt} \bigg\rvert_z & = \frac{\partial u}{\partial t} \bigg\rvert_z - \zeta \rvert_z v + \frac{1}{2}\frac{\partial (u^2 + v^2)}{\partial x}\bigg\rvert_z + w \frac{\partial u}{\partial z} \\

& = \frac{\partial u}{\partial t} \bigg\rvert_z + \left(\frac{\partial u}{\partial y}\bigg\rvert_z - \frac{\partial v}{\partial x}\bigg\rvert_z \right)  v + \frac{1}{2}\frac{\partial (u^2 + v^2)}{\partial x}\bigg\rvert_z + w \frac{\partial u}{\partial z} \\

& = \frac{\partial u}{\partial t} \bigg\rvert_z + \bigg(\underbrace{\frac{\partial u}{\partial y}\bigg\rvert_r - \frac{\partial v}{\partial x}\bigg\rvert_r}_{- \zeta|_r}  - \frac{1}{\sigma} \frac{\partial u}{\partial r} \frac{\partial z}{\partial y} + \frac{1}{\sigma}  \frac{\partial v}{\partial r} \frac{\partial z}{\partial x}   \bigg)  v + \frac{1}{2}\frac{\partial (u^2 + v^2)}{\partial x}\bigg\rvert_z + w \frac{\partial u}{\partial z} \\

& = \frac{\partial u}{\partial t} \bigg\rvert_z - \zeta\rvert_r v - \frac{1}{\sigma} \left(\frac{\partial u}{\partial r} \frac{\partial z}{\partial y} - \frac{\partial v}{\partial r} \frac{\partial z}{\partial x}   \right)  v + \frac{1}{2}\frac{\partial (u^2 + v^2)}{\partial x}\bigg\rvert_z + w \frac{\partial u}{\partial z} \\

& = \frac{\partial u}{\partial t} \bigg\rvert_z - \zeta\rvert_r v - \frac{1}{\sigma} \left(\frac{\partial u}{\partial r} \frac{\partial z}{\partial y} - \frac{\partial v}{\partial r} \frac{\partial z}{\partial x}   \right)  v + \frac{1}{2}\frac{\partial (u^2 + v^2)}{\partial x}\bigg\rvert_r - \frac{1}{2\sigma} \frac{\partial (u^2 + v^2)}{\partial r}\frac{\partial z}{\partial x} + w \frac{\partial u}{\partial z} 
\end{align}
```
Combining all terms divided by $\sigma$ we obtain
```math
\begin{align}
\frac{D u}{Dt} \bigg\rvert_z & =  \frac{\partial u}{\partial t} \bigg\rvert_z - \zeta\rvert_r v + \frac{\partial K}{\partial x}\bigg\rvert_r +
\frac{1}{\sigma} \left( w \frac{\partial u}{\partial r} - v\frac{\partial u}{\partial r} \frac{\partial z}{\partial y}  + v\frac{\partial v}{\partial r} \frac{\partial z}{\partial x}    - u \frac{\partial u}{\partial r}\frac{\partial z}{\partial x}- v \frac{\partial v}{\partial r}\frac{\partial z}{\partial x}\right)  \\

& =  \frac{\partial u}{\partial t} \bigg\rvert_z - \zeta\rvert_r v + \frac{\partial K}{\partial x}\bigg\rvert_r +
\frac{1}{\sigma} \left( w \frac{\partial u}{\partial r} - v \frac{\partial u}{\partial r} \frac{\partial z}{\partial y}  - u \frac{\partial u}{\partial r}\frac{\partial z}{\partial x}\right)  \\

& =  \frac{\partial u}{\partial t} \bigg\rvert_z - \zeta\rvert_r v + \frac{\partial K}{\partial x}\bigg\rvert_r +
\frac{1}{\sigma} \left( w - v \frac{\partial z}{\partial y}  - u \frac{\partial z}{\partial x}\right)  \frac{\partial u}{\partial r}  \\
\end{align}
```
Once again, 
```math
\omega + \frac{\partial z}{\partial t}\bigg\rvert_r =  w - v \frac{\partial z}{\partial y}  - u \frac{\partial z}{\partial x}
```
So that it is possible to write
```math
\begin{align}
\frac{D u}{Dt} \bigg\rvert_z &  =  \frac{\partial u}{\partial t} \bigg\rvert_z - \zeta\rvert_r v + \frac{\partial K}{\partial x}\bigg\rvert_r +
\frac{1}{\sigma} \left( \omega + \frac{\partial z}{\partial t}\bigg\rvert_r \right)  \frac{\partial u}{\partial r}  \\
\end{align}
```
As done above for the tracer, the last term on the right-hand side, using the chain rule for the time derivative yields
```math
\frac{1}{\sigma} \left( \omega + \frac{\partial z}{\partial t}\bigg\rvert_r \right)  \frac{\partial u}{\partial r} = \frac{\omega}{\sigma}\frac{\partial u}{\partial r} + \frac{\partial u}{\partial t}\bigg\rvert_r - \frac{\partial u}{\partial t}\bigg\rvert_z
```
Which completes the derivation of the u-momentum equations in ``r``-coordinates
```math
\frac{D u}{Dt} \bigg\rvert_z  =  \frac{\partial u}{\partial t} \bigg\rvert_r - \zeta\rvert_r v + \frac{\partial K}{\partial x}\bigg\rvert_r + \frac{\omega}{\sigma}\frac{\partial u}{\partial r} 
```
We can further split the vertical advection term into a conservative vertical advection and a horizontal divergence term to obtain
```math
\frac{D u}{Dt} \bigg\rvert_z  =  \frac{\partial u}{\partial t} \bigg\rvert_r - \zeta\rvert_r v + \frac{\partial K}{\partial x}\bigg\rvert_r + \frac{1}{\sigma}\frac{\partial \omega u}{\partial r} - \frac{u}{\sigma} \frac{\partial \omega}{\partial r}
```
Where we can make use of the continuity equation to obtain
```math
\frac{D u}{Dt} \bigg\rvert_z  =  \frac{\partial u}{\partial t} \bigg\rvert_r - \zeta\rvert_r v + \frac{\partial K}{\partial x}\bigg\rvert_r + \frac{1}{\sigma}\frac{\partial \omega u}{\partial r} + \frac{u}{\sigma} \left( \frac{\partial \sigma}{\partial t} + \frac{\partial \sigma u}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial y}\bigg\rvert_{r}\right)
```

### Horizontal pressure gradient

The horizontal pressure gradient $\partial_x p$ can be transformed using the chain rule for spatial derivatives as
```math
\begin{align}
\frac{\partial p}{\partial x}\bigg\rvert_z = & \frac{\partial p}{\partial x}\bigg\rvert_r - \frac{1}{\sigma}\frac{\partial p}{\partial r}\frac{\partial z}{\partial x} \\
\end{align}
```
where using the hydrostatic relation we can write
```math
\begin{equation}
\frac{\partial p}{\partial x}\bigg\rvert_z = \frac{\partial p}{\partial x}\bigg\rvert_r - b \frac{\partial z}{\partial x} 
\end{equation}
```
where the additional term describes the pressure gradient associated with the horizontal tilting of the grid.
The gradient of surface pressure (the free surface) remains unchanged under vertical coordinate transformation
```math
\begin{align}
g \frac{\partial \eta}{\partial x}\bigg\rvert_z & = g \frac{\partial \eta}{\partial x}\bigg\rvert_r - \frac{g}{\sigma} \frac{\partial \eta}{\partial r}\frac{\partial z}{\partial x} \\
& = g \frac{\partial \eta}{\partial x}\bigg\rvert_r
\end{align}
```

