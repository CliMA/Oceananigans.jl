# Generalized vertical coordinate

For `HydrostaticFreeSurfaceModel()`, the user can choose between a `ZCoordinate` and a `ZStar` vertical coordinate.
A `ZStar` vertical coordinate conserves tracers and volume with the grid following the evolution of the free surface in the domain [adcroft2004rescaled](@citet).
To obtain the (discrete) equations evolved  in a general framework where the vertical coordinate is moving, we perform a scaling of the continuous primitive equations to a generalized coordinate ``r(x, y, z, t)``.

We have that:

```math
\begin{alignat}{2}
& \frac{\partial \phi}{\partial s}\bigg\rvert_{z} && =  \frac{\partial \phi}{\partial s}\bigg\rvert_{r} + \frac{\partial \phi}{\partial r}\cdot \frac{\partial r}{\partial s} \\ 
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
& = \frac{\partial u}{\partial x} \bigg\rvert_{r} + \frac{\partial v}{\partial y} \bigg\rvert_{r} - \frac{1}{\sigma} \left( \frac{\partial u}{\partial r} \frac{\partial z}{\partial x} + \frac{\partial v}{\partial y} \frac{\partial z}{\partial y}  - \frac{\partial w}{\partial r} \right) \\ 
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
Here, $w$ is the vertical velocity corresponding to the ``z`` coordinate. We can define a vertical velocity $w_p$ of a point moving with the horizontal velocity along an ``r`` surface 
```math
w_p = \frac{\partial z}{\partial t} \bigg\rvert_r + u \frac{\partial z}{\partial x} +  v \frac{\partial z}{\partial y}
```
Then, the vertical velocity across the ``r`` surfaces is the difference of $w$ and $w_p$
```math
\omega = w - w_p = w - \frac{\partial z}{\partial t} \bigg\rvert_r - u \frac{\partial z}{\partial x} - v \frac{\partial z}{\partial y}
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
\frac{\partial T}{\partial t}\bigg\rvert_{z} + \boldsymbol{\nabla} \cdot \boldsymbol{u}T = \frac{1}{\sigma}\frac{\partial \sigma T}{\partial t}\bigg\rvert_r + \frac{1}{\sigma} \left( \frac{\partial \sigma u T}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v T}{\partial y}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial T \omega}{\partial r}
```
We add vertical diffusion to the RHS to recover the tracer equation
```math
\frac{1}{\sigma}\frac{\partial \sigma T}{\partial t}\bigg\rvert_r + \frac{1}{\sigma} \left( \frac{\partial \sigma u T}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v T}{\partial y}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial T \omega}{\partial r} = \frac{1}{\sigma}\frac{\partial}{\partial r} \left( \kappa \frac{\partial T}{\partial z} \right)
```
### Finite-volume discretization of the tracer equation

We discretize the equation in a finite volume framework
```math
\frac{1}{V_r}\int_{V_r} \frac{1}{\sigma}\frac{\partial \sigma T}{\partial t} + \frac{1}{V_r} \int_{V_r} \left[ \frac{1}{\sigma} \left( \frac{\partial \sigma u T}{\partial x} \bigg\rvert_{r} + \frac{\partial \sigma v T}{\partial y}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial T \omega}{\partial r}\right] \, \mathrm{d}V = \frac{1}{V_r}\int_{V_r} \frac{1}{\sigma}\frac{\partial}{\partial r} \left( \kappa \frac{\partial T}{\partial z} \right) \, \mathrm{d}V
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

