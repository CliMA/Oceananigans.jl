# [Pressure decomposition](@id pressure_decomposition)

In the numerical implementation of the momentum equations in the `NonhydrostaticModel`, the kinematic pressure ``p``
is split into "background" and "dynamic" parts via
```math
    \begin{equation}
    \label{eq:pressure}
    p(\boldsymbol{x}, t) = p_{\text{background}}(\boldsymbol{x}, t) + p'(\boldsymbol{x}, t) \, .
    \end{equation}
```

The background pressure component in \eqref{eq:pressure} is defined so that the vertical
component of its gradient balances the background density field:

```math
    \begin{align}
    \partial_z p_{\text{total hydrostatic}} & = - g \left ( 1 + \frac{\rho_*}{\rho_0} \right ) \, ,
    \end{align}
```

Above, we use the notation introduced in the [Boussinesq approximation](@ref boussinesq_approximation)
section.

Optionally, we may further decompose the dynamic pressure perturbation ``p'`` into
a "hydrostatic anomaly" and "nonhydrostatic" part:
```math
    \begin{align}
    p'(\boldsymbol{x}, t) = p_{\rm{hyd}}(\boldsymbol(x), t) + p_{\rm{non}}(\boldsymbol{x}, t) \, ,
    \end{align}
```

where

```math
    \begin{align}
    \partial_z p_{\rm{hyd}} \equiv \underbrace{- g \frac{\rho'}{\rho_0}}_{= b} \, .
    \end{align}
```

With this pressure decomposition, the kinematic pressure gradient that appears in the momentum equations
(after we've employed the the [Boussinesq approximation](@ref boussinesq_approximation)) becomes

```math
    \begin{align}
    \boldsymbol{\nabla} p &= - g \frac{\rho}{\rho_0} \hat {\boldsymbol{z}} + \boldsymbol{\nabla} p'
                          &= - g \frac{\rho}{\rho_0} \hat {\boldsymbol{z}} + \boldsymbol{\nabla} p_{\rm{non}} + \boldsymbol{\nabla}_h p_{\rm{hyd}} \, .
    \end{align}
```

where ``\boldsymbol{\nabla}_h \equiv \boldsymbol{\hat x} \partial_x +  \boldsymbol{\hat y} \partial_y``.

