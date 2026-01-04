# [Generalized vertical coordinates](@id generalized_vertical_coordinates)

Often the numerics benefit from using a vertical coordinate that is different than the physical
vertical coordinate ``z`` and which, in general, may even be moving (vary in time).

## Coordinate systems and notation

We distinguish between two coordinate systems:

- **Physical coordinates** ``(x, y, z)``: The locally-Cartesian coordinates in which the equations of motion are naturally expressed.
- **Computational coordinates** ``(\xi, \eta, r)``: The transformed coordinates used for discretization, where ``\xi`` and ``\eta`` are horizontal computational coordinates and ``r`` is the generalized vertical coordinate.

!!! note "Notation for the free surface"
    To avoid confusion between the computational ``\eta``-coordinate and the free surface displacement,
    we denote the **free surface** as ``\eta_{\rm fs}(\xi, \eta, t)`` in this documentation.
    In the code, the free surface is typically stored in a variable named `η`.

We denote any such generalized vertical coordinate that evolves with space and time as ``r(\xi, \eta, z, t)``.
The generalized vertical coordinate must vary monotonically with ``z``.

For the `ZStarCoordinate` (zee-star) coordinate, the mapping from ``r`` to ``z`` is affine in ``r``:
```math
z(\xi, \eta, r, t) = \eta_{\rm fs}(\xi, \eta, t) + \sigma(\xi, \eta, t) \, r
```
with the "specific thickness"
```math
\sigma(\xi, \eta, t) = \frac{H(\xi, \eta) + \eta_{\rm fs}(\xi, \eta, t)}{H(\xi, \eta)} = 1 + \frac{\eta_{\rm fs}}{H}
```
where ``H(\xi, \eta)`` is the static column depth (distance from the resting free surface to the bottom).

## Chain rules for coordinate transformation

Transforming the equations of motion (including their discrete analogues) in the generalized coordinate
involves a series of chain rules for partial differentiation. We have that for any field ``\phi``:

```math
\begin{align}
    \frac{\partial \phi}{\partial z} & = \frac{\partial r}{\partial z} \frac{\partial \phi}{\partial r} \label{dphidz} \\
    \frac{\partial \phi}{\partial r} & = \frac{\partial z}{\partial r} \frac{\partial \phi}{\partial z}
\end{align}
```

The quantity ``\partial z/\partial r`` plays a central role in what follows. We refer to it as "specific thickness" and denote it ``\sigma``:
```math
\sigma \equiv \frac{\partial z}{\partial r} \bigg\rvert_{\xi, \eta, t} = \left(\frac{\partial r}{\partial z} \bigg\rvert_{\xi, \eta, t}\right)^{-1}
```
where the subscripts next to ``\vert`` denote the quantities that remain constant in the differentiation.

The chain rules for differentiation with respect to ``\xi``, ``\eta``, or ``t`` become:

```math
\begin{align}
    \frac{\partial \phi}{\partial s} \bigg\rvert_{z} & = \frac{\partial \phi}{\partial s}\bigg\rvert_{r} + \frac{\partial \phi}{\partial r} \frac{\partial r}{\partial s} \label{dphids}
\end{align}
```

The horizontal spatial derivatives of the ``r``-coordinate are then rewritten, e.g.,
```math
\begin{equation}
    \frac{\partial r}{\partial \xi} \bigg\rvert_{\eta, z, t} = - \frac{\partial z}{\partial \xi} \bigg\rvert_{\eta, r, t} \frac{1}{\sigma} \label{drdxi}
\end{equation}
```
so that the chain rule \eqref{dphids} above becomes
```math
\begin{align}
    \frac{\partial \phi}{\partial \xi} \bigg\rvert_{z} & = \frac{\partial \phi}{\partial \xi}\bigg\rvert_{r} - \frac{1}{\sigma}\frac{\partial \phi}{\partial r} \frac{\partial z}{\partial \xi} \label{dphidxi}
\end{align}
```
Similarly, we get equivalent expressions for ``\eta`` derivatives as in \eqref{drdxi} and \eqref{dphidxi}.

## Implementation in Oceananigans

In Oceananigans, the discrete operators are organized as follows:

- **Difference operators** ``\delta_\xi``, ``\delta_\eta``, ``\delta_r``: These compute finite differences between neighboring grid points and are **unchanged** by the coordinate transformation. They operate purely in computational space.

- **Derivative operators** ``\partial_\xi``, ``\partial_\eta``, ``\partial_r``: For grids using `ZStarVerticalCoordinate`, the horizontal derivative operators ``\partial_\xi`` and ``\partial_\eta`` are redefined to include the chain-rule correction term:
```math
\partial_\xi \phi \rvert_z = \partial_\xi \phi \rvert_r - (\partial_\xi z \rvert_r)(\partial_z \phi)
```
```math
\partial_\eta \phi \rvert_z = \partial_\eta \phi \rvert_r - (\partial_\eta z \rvert_r)(\partial_z \phi)
```

- **Vertical derivative** ``\partial_z``: The vertical derivative already includes the stretching factor because it uses ``\Delta z = \sigma \Delta r``:
```math
\partial_z \phi = \frac{1}{\sigma} \partial_r \phi
```

This design means that physics code can use ``\partial_\xi``, ``\partial_\eta``, ``\partial_z`` and obtain the correct physical derivatives without needing to explicitly handle the coordinate transformation.

## Mass conservation

With the [Boussinesq approximation](@ref boussinesq_approximation), the mass conservation reduces to the flow being divergence-less, i.e., ``\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{v} = \boldsymbol{\nabla}_h \boldsymbol{\cdot} \boldsymbol{u} + \partial_z w = 0``.

Using the chain rules above, the divergence of the flow in ``r``-coordinates becomes:
```math
\begin{align}
    \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{v} & = \frac{\partial u}{\partial \xi} \bigg\rvert_{z} + \frac{\partial v}{\partial \eta} \bigg\rvert_{z} + \frac{\partial w}{\partial z} \nonumber \\
    & = \frac{\partial u}{\partial \xi} \bigg\rvert_{r} + \frac{\partial v}{\partial \eta} \bigg\rvert_{r} - \frac{1}{\sigma} \left( \frac{\partial u}{\partial r} \frac{\partial z}{\partial \xi} + \frac{\partial v}{\partial r} \frac{\partial z}{\partial \eta} - \frac{\partial w}{\partial r} \right) \nonumber \\
    & = \frac{1}{\sigma} \left( \frac{\partial \sigma u}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial \eta} \bigg\rvert_{r} - u \frac{\partial \sigma}{\partial \xi} \bigg\rvert_{r} - v \frac{\partial \sigma}{\partial \eta} \bigg\rvert_{r} \right)- \frac{1}{\sigma} \left( \frac{\partial u}{\partial r} \frac{\partial z}{\partial \xi} + \frac{\partial v}{\partial \eta} \frac{\partial z}{\partial \eta} - \frac{\partial w}{\partial r} \right)
\end{align}
```
We can rewrite ``\partial_\xi \sigma \rvert_r = \partial_r(\partial_\xi z)`` and similarly for the ``\eta`` direction. After a bit of reordering the above yields
```math
\begin{equation}
    \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{v} = \frac{1}{\sigma} \left( \frac{\partial \sigma u}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial \eta}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial}{\partial r} \left( u \frac{\partial z}{\partial \xi} + v \frac{\partial z}{\partial \eta} + w \right) \label{div1}
\end{equation}
```
Note that ``w`` above is the vertical velocity referenced to the ``z`` coordinate.
The vertical velocity ``w_r`` of the ``r`` surface referenced to the ``z`` coordinate as
```math
w_r \equiv \frac{\partial z}{\partial t} \bigg\rvert_r + u \frac{\partial z}{\partial \xi} + v \frac{\partial z}{\partial \eta}
```
Then, the vertical velocity across the ``r`` surfaces is the difference between ``w`` and ``w_r``
```math
\begin{align}
    \omega & \equiv w - w_r \nonumber \\
           & = w - \frac{\partial z}{\partial t} \bigg\rvert_r - u \frac{\partial z}{\partial \xi} - v \frac{\partial z}{\partial \eta} \label{def_omega}
\end{align}
```
With the definition of ``\omega`` in \eqref{div1} we get
```math
\begin{align}
    \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{v} & = \frac{1}{\sigma} \left( \frac{\partial \sigma u}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial \eta}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial}{\partial r} \left( \omega + \frac{\partial z}{\partial t}\bigg\rvert_r \right) \nonumber \\
    & = \frac{1}{\sigma} \left( \frac{\partial \sigma u}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial \eta}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial \omega}{\partial r} + \frac{1}{\sigma} \frac{\partial \sigma}{\partial t}
\end{align}
```
which implies that the mass conservation is equivalent to:
```math
\begin{equation}
    \frac{\partial \sigma}{\partial t} + \frac{\partial \sigma u}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial \eta}\bigg\rvert_{r} + \frac{\partial \omega}{\partial r} = 0 \label{massconservationr}
\end{equation}
```

## Tracer equation

The evolution equation for a tracer ``c``, which also includes vertical diffusion, reads
```math
\begin{equation}
    \frac{\partial c}{\partial t}\bigg\rvert_{z} + \boldsymbol{\nabla} \boldsymbol{\cdot} (\boldsymbol{u} c) = \frac{\partial}{\partial z} \left( \kappa \frac{\partial c}{\partial z} \right) \label{tracereq}
\end{equation}
```
Using the same procedure we followed for the continuity equation, the left-hand-side of \eqref{tracereq} yields:
```math
\begin{align}
    \frac{\partial c}{\partial t}\bigg\rvert_{z} & + \boldsymbol{\nabla} \boldsymbol{\cdot} (\boldsymbol{u} c) = \nonumber \\
    & = \frac{\partial c}{\partial t}\bigg\rvert_{z} + \frac{1}{\sigma} \left( \frac{\partial \sigma u c}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v T}{\partial \eta}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial}{\partial r}\left(c \omega + c \frac{\partial z}{\partial t}\bigg\rvert_r \right) \nonumber \\
    & = \frac{\partial c}{\partial t}\bigg\rvert_{z} + \frac{1}{\sigma} \left( \frac{\partial \sigma u c}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v c}{\partial \eta}\bigg\rvert_{r} \right) + \frac{c}{\sigma} \left( \frac{\partial \omega}{\partial r} + \frac{\partial \sigma}{\partial t}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \left( \omega + \frac{\partial z}{\partial t}\bigg\rvert_r \right)\frac{\partial c}{\partial r} \nonumber\\
    & = \frac{\partial c}{\partial t}\bigg\rvert_{z} + \frac{1}{\sigma} \left( \frac{\partial \sigma u c}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v c}{\partial \eta}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial \omega c}{\partial r} + \frac{c}{\sigma} \frac{\partial \sigma}{\partial t}\bigg\rvert_{r} + \frac{1}{\sigma} \frac{\partial z}{\partial t}\bigg\rvert_r \frac{\partial c}{\partial r}
\end{align}
```
We recover the time derivative of the tracer at constant ``r`` by rewriting the last term using the chain rule for a time derivatives:
```math
\frac{1}{\sigma}  \frac{\partial z}{\partial t}\bigg\rvert_r \frac{\partial c}{\partial r} = \frac{\partial r}{\partial t} \frac{\partial c}{\partial r} = \frac{\partial c}{\partial t}\bigg\rvert_{r} - \frac{\partial c}{\partial t}\bigg\rvert_{z}
```
As such, the left-hand-side of \eqref{tracereq} can be rewritten in ``r``-coordinates as
```math
    \frac{1}{\sigma}\frac{\partial \sigma c}{\partial t}\bigg\rvert_r + \frac{1}{\sigma} \left( \frac{\partial \sigma u c}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v c}{\partial \eta}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial \omega c}{\partial r}
```
Including the vertical diffusion to the right-hand side we recover the tracer equation:
```math
\begin{equation}
    \frac{1}{\sigma}\frac{\partial \sigma c}{\partial t} \bigg\rvert_r + \frac{1}{\sigma} \left( \frac{\partial \sigma u c}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v c}{\partial \eta}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial c \omega}{\partial r} = \frac{1}{\sigma}\frac{\partial}{\partial r} \left( \kappa \frac{\partial c}{\partial z} \right) \label{tracerrcoord}
\end{equation}
```

## Momentum equations in vector invariant form

The [horizontal momentum equations under the hydrostatic approximation](@ref hydrostatic_free_surface_model) read
```math
\begin{equation}
    \frac{\mathrm{D} \boldsymbol{u}}{\mathrm{D}t} \bigg\rvert_z + f \hat{\boldsymbol{z}} \times \boldsymbol{u} = - \boldsymbol{\nabla} p \rvert_z - g \boldsymbol{\nabla} \eta_{\rm fs} \rvert_z + \frac{\partial}{\partial z}  \left(\nu \frac{\partial \boldsymbol{u}_h}{\partial z}\right) \label{momentumh}
\end{equation}
```
where ``\boldsymbol{u} = u \hat{\boldsymbol{x}} + v \hat{\boldsymbol{y}}`` is the horizontal velocity, ``\boldsymbol{v} = \boldsymbol{u} + w \hat{\boldsymbol{z}}`` is the three-dimensional velocity, and ``\mathrm{D} / \mathrm{D}t \equiv \partial_t + \boldsymbol{v \cdot \nabla}`` is the material derivative.

The above is complemented by the hydrostatic relation
```math
\begin{equation}
    \frac{\partial p}{\partial z} = b \label{hydrostatic}
\end{equation}
```

Of the above, the Coriolis term is independent of the vertical frame of reference and the viscous stress is treated similarly to the diffusion of a tracer. In this derivation we focus on:
1. the hydrostatic relation \eqref{hydrostatic},
2. the material derivative in the momentum equation \eqref{momentumh}, and
3. the horizontal pressure gradient terms in \eqref{momentumh}.

### Hydrostatic relation
Using the definition \eqref{dphidz} of the ``z``-derivative in ``r``-coordinates
```math
\begin{equation}
    \frac{\partial p}{\partial r} = \sigma b \label{hydrostaticrcoord}
\end{equation}
```

### Material derivative in vector invariant form

We set out to transform in ``r``-coordinates the material derivative of the horizontal velocity in vector invariant form
```math
\frac{\mathrm{D} \boldsymbol{u}}{\mathrm{D} t} \bigg\rvert_z = \frac{\partial \boldsymbol{u}}{\partial t} \bigg\rvert_z + \zeta \hat{\boldsymbol{z}} \times \boldsymbol{u} + \boldsymbol{\nabla}_h K + w \frac{\partial \boldsymbol{u}}{\partial z}
```
where ``\zeta = \partial_\xi v - \partial_\eta u`` is the vertical vorticity, and ``K \equiv (u^2 + v^2)/2`` is the horizontal kinetic energy.

Here, we focus on the ``u`` component of the velocity; the derivation of the ``v`` component follows the same steps.
Thus, we are transforming
```math
\begin{align*}
    \frac{\mathrm{D}u}{\mathrm{D}t} \bigg\rvert_z & = \frac{\partial u}{\partial t} \bigg\rvert_z - \zeta \rvert_z v + \frac{\partial K}{\partial \xi}\bigg\rvert_z + w \frac{\partial u}{\partial z} \\

    & = \frac{\partial u}{\partial t} \bigg\rvert_z - \left(\frac{\partial v}{\partial \xi}\bigg\rvert_z - \frac{\partial u}{\partial \eta}\bigg\rvert_z \right) v + \frac{\partial K}{\partial \xi}\bigg\rvert_z + w \frac{\partial u}{\partial z} \\

    & = \frac{\partial u}{\partial t} \bigg\rvert_z - \bigg(\underbrace{\frac{\partial v}{\partial \xi}\bigg\rvert_r - \frac{\partial u}{\partial \eta}\bigg\rvert_r}_{\zeta|_r} - \frac{1}{\sigma} \frac{\partial v}{\partial r} \frac{\partial z}{\partial \xi} + \frac{1}{\sigma} \frac{\partial u}{\partial r} \frac{\partial z}{\partial \eta} \bigg) v + \frac{\partial K}{\partial \xi}\bigg\rvert_r - \frac{1}{\sigma} \frac{\partial K}{\partial r}\frac{\partial z}{\partial \xi} + w \frac{\partial u}{\partial z}
\end{align*}
```
Above, we utilized \eqref{dphids} and \eqref{drdxi} repeatedly, e.g., for ``\partial_\eta u \rvert_z``, ``\partial_\xi v \rvert_z``, and ``\partial_\xi K \rvert_z``.
Further expanding ``\partial_r K = u \partial_r u + v \partial_r v``, a few terms cancel out and we end up with:
```math
\begin{align}
    \frac{\mathrm{D}u}{\mathrm{D}t} \bigg\rvert_z
    & = \frac{\partial u}{\partial t} \bigg\rvert_z - \zeta\rvert_r v + \frac{\partial K}{\partial \xi}\bigg\rvert_r + \frac{1}{\sigma} \left( w - v \frac{\partial z}{\partial \eta} - u \frac{\partial z}{\partial \xi}\right)  \frac{\partial u}{\partial r} \label{expr1}
\end{align}
```
Using the definition of ``\omega`` in \eqref{def_omega}, we can rewrite \eqref{expr1} as
```math
\begin{align}
    \frac{\mathrm{D}u}{\mathrm{D}t} \bigg\rvert_z & = \frac{\partial u}{\partial t} \bigg\rvert_z - \zeta\rvert_r v + \frac{\partial K}{\partial \xi}\bigg\rvert_r + \frac{1}{\sigma} \left( \omega + \frac{\partial z}{\partial t}\bigg\rvert_r \right)  \frac{\partial u}{\partial r}
\end{align}
```
As done above for the tracer, the last term on the right-hand side, using the chain rule for the time derivative yields
```math
\frac{1}{\sigma} \left( \omega + \frac{\partial z}{\partial t}\bigg\rvert_r \right)  \frac{\partial u}{\partial r} = \frac{\omega}{\sigma}\frac{\partial u}{\partial r} + \frac{\partial u}{\partial t}\bigg\rvert_r - \frac{\partial u}{\partial t}\bigg\rvert_z
```
Which completes the derivation of the ``u``-momentum equations in ``r``-coordinates
```math
\frac{\mathrm{D}u}{\mathrm{D}t} \bigg\rvert_z = \frac{\partial u}{\partial t} \bigg\rvert_r - \zeta\rvert_r v + \frac{\partial K}{\partial \xi}\bigg\rvert_r + \frac{\omega}{\sigma}\frac{\partial u}{\partial r}
```
We can further split the vertical advection term into a conservative vertical advection and a horizontal divergence term:
```math
\frac{\mathrm{D}u}{\mathrm{D}t} \bigg\rvert_z = \frac{\partial u}{\partial t} \bigg\rvert_r - \zeta\rvert_r v + \frac{\partial K}{\partial \xi}\bigg\rvert_r + \frac{1}{\sigma}\frac{\partial \omega u}{\partial r} - \frac{u}{\sigma} \frac{\partial \omega}{\partial r}
```
and last using the continuity equation \eqref{massconservationr} to obtain
```math
\begin{equation}
    \frac{\mathrm{D}u}{\mathrm{D}t} \bigg\rvert_z = \frac{\partial u}{\partial t} \bigg\rvert_r - \zeta\rvert_r v + \frac{\partial K}{\partial \xi}\bigg\rvert_r + \frac{1}{\sigma}\frac{\partial \omega u}{\partial r} + \frac{u}{\sigma} \left( \frac{\partial \sigma}{\partial t} + \frac{\partial \sigma u}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial \eta}\bigg\rvert_{r}\right)
\end{equation}
```

### Horizontal pressure gradient

The horizontal pressure gradients, e.g., ``\partial_\xi p``, can be transformed using the chain rule \eqref{dphidxi}:
```math
\begin{equation}
    \frac{\partial p}{\partial \xi} \bigg\rvert_z = \frac{\partial p}{\partial \xi}\bigg\rvert_r - \frac{1}{\sigma}\frac{\partial p}{\partial r}\frac{\partial z}{\partial \xi}
\end{equation}
```
and combined with the hydrostatic relation \eqref{hydrostaticrcoord}:
```math
\begin{equation}
    \frac{\partial p}{\partial \xi}\bigg\rvert_z = \frac{\partial p}{\partial \xi}\bigg\rvert_r - b \frac{\partial z}{\partial \xi}
\end{equation}
```
The last term on the right hand side above describes the pressure gradient associated with the horizontal tilting of the grid.

!!! note "Absorbed into derivative operators"
    In Oceananigans, this "grid slope" correction term is automatically included when using the
    ``\partial_\xi`` and ``\partial_\eta`` derivative operators on grids with `ZStarVerticalCoordinate`.
    Physics code does not need to add this term separately.

Similarly, the gradient of the free surface transforms to
```math
\begin{align}
    \frac{\partial \eta_{\rm fs}}{\partial \xi} \bigg\rvert_z & = \frac{\partial \eta_{\rm fs}}{\partial \xi}\bigg\rvert_r - \frac{1}{\sigma} \frac{\partial \eta_{\rm fs}}{\partial r}\frac{\partial z}{\partial \xi}
\end{align}
```


## Finite volume discretization

It is useful to describe how the above equations translate into discrete form in a finite volume staggered C-grid framework.

To do so, we integrate over the cell volume ``V_r = \Delta \xi \Delta \eta \Delta r`` remembering that in the discrete ``\Delta z = \sigma \Delta r``.
Indices `i`, `j`, `k` below correspond to the ``\xi``, ``\eta``, and the vertical directions respectively.

### Mass conservation

The mass conservation \eqref{massconservationr} gives:
```math
\begin{equation}
    \frac{1}{V_r}\int_{V_r} \frac{\partial \sigma}{\partial t} \, \mathrm{d}V + \frac{1}{V_r} \int_{V_r} \left(\frac{\partial \sigma u}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v}{\partial \eta}\bigg\rvert_{r} + \frac{\partial \omega}{\partial r}\right) \, \mathrm{d}V = 0 \label{massdiscr1}
\end{equation}
```
Using the notation for cell-averages ``\overline{\phi} \equiv V_r^{-1} \int_{V_r} \phi \,\mathrm{d}V`` and also the divergence theorem,
we can rewrite \eqref{massdiscr1} to:
```math
\begin{equation}
    \frac{\partial \overline{\sigma}}{\partial t} + \frac{1}{\Delta \xi \Delta \eta \Delta r} \left( \Delta \eta \Delta r \, \sigma u \rvert_{i-1/2}^{i+1/2} + \Delta \xi \Delta r \, \sigma v \rvert_{j-1/2}^{j+1/2} \right ) + \frac{\overline{\omega}_{k+1/2} - \overline{\omega}_{k-1/2}}{\Delta r} = 0 \label{massdiscr2}
\end{equation}
```
We use \eqref{massdiscr2} diagnose the vertical velocity (in ``r`` space) given the grid velocity and the horizontal velocity divergence:
```math
\overline{\omega}_{k+1/2} = \overline{\omega}_{k-1/2} + \Delta r \frac{\partial \overline{\sigma}}{\partial t} + \frac{1}{A_z} \left( \mathcal{U}\rvert_{i-1/2}^{i+1/2} + \mathcal{V}\rvert_{j-1/2}^{j+1/2} \right )
```
where ``\mathcal{U} = A_\xi \, u``, ``\mathcal{V} = A_\eta \, v``, ``A_\xi = \Delta \eta \, \Delta z``, ``A_\eta = \Delta \xi \, \Delta z``, and ``A_z = \Delta \xi \, \Delta \eta``.

### Tracer equation

The tracer equation \eqref{tracerrcoord} in discrete form becomes:
```math
\frac{1}{V_r}\int_{V_r} \frac{1}{\sigma}\frac{\partial \sigma c}{\partial t} \, \mathrm{d}V + \frac{1}{V_r} \int_{V_r} \left[ \frac{1}{\sigma} \left( \frac{\partial \sigma u c}{\partial \xi} \bigg\rvert_{r} + \frac{\partial \sigma v c}{\partial \eta}\bigg\rvert_{r} \right) + \frac{1}{\sigma} \frac{\partial \omega c}{\partial r}\right] \, \mathrm{d}V = \frac{1}{V_r}\int_{V_r} \frac{1}{\sigma}\frac{\partial}{\partial r} \left( \kappa \frac{\partial c}{\partial z} \right) \, \mathrm{d}V
```
leading to
```math
\frac{1}{\sigma}\frac{\partial \sigma \overline{c}}{\partial t} + \frac{\mathcal{U}c \rvert_{i-1/2}^{i+1/2} + \mathcal{V}c \rvert_{j-1/2}^{j+1/2} + \mathcal{W} c \rvert_{k-1/2}^{k+1/2}}{V} = \frac{1}{V} \left(\mathcal{K} \frac{\partial c}{\partial z}\bigg\rvert_{k-1/2}^{k+1/2} \right)
```
where ``V = \sigma V_r = \Delta \xi \, \Delta \eta \, \Delta z``, ``\mathcal{W} = A_z \, \omega``, and ``\mathcal{K} = A_z \, \kappa``.

For an explicit formulation of the diffusive fluxes and a time-discretization using forward Euler scheme, imply:
```math
\begin{equation}
    c^{n+1} = \frac{\sigma^n}{\sigma^{n+1}}\left(c^n + \Delta t \, G^n \right)
\end{equation}
```
where ``G^n`` is tendency computed on the `z`-grid.

Note that in case of a multi-step method, e.g., second-order Adams-Bashforth, the grid at different time-steps must be accounted for, and the time-discretization becomes
```math
\begin{equation}
    c^{n+1} = \frac{1}{\sigma^{n+1}} \left[\sigma^n c^n + \Delta t \left(\frac{3}{2} \sigma^n G^n - \frac{1}{2} \sigma^{n-1} G^{n-1}\right)\right]
\end{equation}
```
For this reason, we store the tendencies pre-multiplied by ``\sigma`` at their current time-level.
In case of an implicit discretization of the diffusive fluxes we first compute ``c^{n+1}`` as in the above equation (where ``G^n`` does not contain the diffusive fluxes).
Then the implicit step is done on a `z`-grid as if the grid was static, using the grid at ``n+1`` which includes ``\sigma^{n+1}``.
