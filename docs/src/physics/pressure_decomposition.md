# [Pressure decomposition](@id pressure_decomposition)

In the implementation of the momentum equations, the kinematic pressure ``p`` is split into 
"hydrostatic" and "non-hydrostatic" parts via
```math
    \begin{equation}
    \label{eq:pressure}
    p(\boldsymbol{x}, t) = p_{\text{total hydrostatic}}(\boldsymbol{x}, t) + p_{\rm{non}}(\boldsymbol{x}, t) \, .
    \end{equation}
```

The hydrostatic pressure component in \eqref{eq:pressure} is defined so that
```math
    \begin{align}
    \partial_z p_{\text{total hydrostatic}} & = - g \left ( 1 + \frac{\rho_*}{\rho_0} + \frac{\rho'}{\rho_0} \right ) \, .
    \end{align}
```

We can further split the hydrostatic pressure component into
```math
    \begin{align}
    p_{\text{total hydrostatic}}(\boldsymbol{x}, t) = p_{*}(z) + p_{\rm{hyd}}(\boldsymbol{x}, t) \, ,
    \end{align}
```

i.e., a component that only varies in ``z`` (``p_*``) and a "hydrostatic anomaly" (``p_{\rm{hyd}}``) defined
so that

```math
    \begin{align}
    \partial_z p_{*} & = - g \left ( 1 + \frac{\rho_*}{\rho_0} \right ) \, ,\\
    \partial_z p_{\rm{hyd}} & = - g \frac{\rho'}{\rho_0} = b \, .
    \end{align}
```

Doing so, the gradient of the kinematic pressure becomes:

```math
    \begin{align}
    \boldsymbol{\nabla} p & = \boldsymbol{\nabla} p_{\rm{non}} + \boldsymbol{\nabla}_h p_{\rm{hyd}} + \partial_z p_{*} \boldsymbol{\hat z} + \partial_z p_{\rm{hyd}} \boldsymbol{\hat z}\, ,
    \end{align}
```

where ``\boldsymbol{\nabla}_h \equiv \boldsymbol{\hat x} \partial_x +  \boldsymbol{\hat y} \partial_y``
is the horizontal gradient.

Under this pressure decomposition, the pressure gradient that appears in the momentum equations combines with
the gravity force to give:

```math
    \begin{align}
    \boldsymbol{\nabla} p + g \frac{\rho}{\rho_0} \hat {\boldsymbol{z}} = \boldsymbol{\nabla} p_{\rm{non}} + \boldsymbol{\nabla}_h p_{\rm{hyd}} \, .
    \end{align}
```

Mathematically, the non-hydrostatic pressure ``p_{\rm{non}}`` enforces the incompressibility constraint.
