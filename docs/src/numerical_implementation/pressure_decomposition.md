# Pressure decomposition

In the numerical implementation of the momentum equations, the kinematic pressure ``p`` 
is split into "hydrostatic anomaly" and "non-hydrostatic" parts via
```math
    \begin{equation}
    \label{eq:pressure}
    p(\boldsymbol{x}, t) = p_{\rm{hyd}}(\boldsymbol{x}, t) + p_{\rm{non}}(\boldsymbol{x}, t) \, .
    \end{equation}
```
The anomalous hydrostatic component of the kinematic pressure is defined by 
```math
    \begin{align}
    \label{eq:hydrostaticpressure}
    \partial_z p_{\rm{hyd}} \equiv -b \, ,
    \end{align}
```
such that the sum of the kinematic pressure and buoyancy perturbation becomes
```math
    \begin{align}
    -\boldsymbol{\nabla} p + b \boldsymbol{\hat z} = 
        - \boldsymbol{\nabla} p_{\rm{non}}
        - \boldsymbol{\nabla}_h p_{\rm{hyd}} \, ,
    \end{align}
```
where ``\boldsymbol{\nabla}_h \equiv \partial_x \boldsymbol{\hat x} + \partial_y \boldsymbol{\hat y}`` 
is the horizontal gradient. The hydrostatic pressure anomaly is so named because the "total" 
hydrostatic pressure contains additional components:
```math
\begin{align}
\partial_z p_{\text{total hydrostatic}} & = - g \left ( 1 + \frac{\rho_*}{\rho_0} + \frac{\rho'}{\rho_0} \right ) \, , \\
                                           & = \partial_z p_{\rm{hyd}} - g \left ( 1 + \frac{\rho_*}{\rho_0} \right ) \, .
\end{align}
```
Under this pressure decomposition the pressure gradient that appears in the momentum equations becomes
```math
   \boldsymbol{\nabla} p \mapsto \boldsymbol{\nabla} p_{\rm{non}} + \boldsymbol{\nabla}_h p_{\rm{hyd}}\, .
```
Mathematically, the non-hydrostatic pressure ``p_{\rm{non}}`` enforces the incompressibility constraint.
