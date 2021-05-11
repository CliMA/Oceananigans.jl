# Pressure decomposition

In the numerical implementation of the momentum equations, the kinematic potential ``\phi`` 
is split into "hydrostatic anomaly" and "non-hydrostatic" parts via
```math
    \begin{equation}
    \label{eq:pressure}
    \phi(\boldsymbol{x}, t) = \phi_{\rm{hyd}}(\boldsymbol{x}, t) + \phi_{\rm{non}}(\boldsymbol{x}, t) \, .
    \end{equation}
```
The anomalous hydrostatic component of the kinematic potential is defined by 
```math
    \begin{align}
    \label{eq:hydrostaticpressure}
    \partial_z \phi_{\rm{hyd}} \equiv -b \, ,
    \end{align}
```
such that the sum of the kinematic potential and buoyancy perturbation becomes
```math
    \begin{align}
    -\boldsymbol{\nabla} \phi + b \boldsymbol{\hat z} = 
        - \boldsymbol{\nabla} \phi_{\rm{non}}
        - \boldsymbol{\nabla}_h \phi_{\rm{hyd}} \, ,
    \end{align}
```
where ``\boldsymbol{\nabla}_h \equiv \partial_x \boldsymbol{\hat x} + \partial_y \boldsymbol{\hat y}`` 
is the horizontal gradient. The hydrostatic pressure anomaly is so named because the "total" 
hydrostatic pressure contains additional components:
```math
\begin{align}
\partial_z \phi_{\text{total hydrostatic}} & = - g \left ( 1 + \frac{\rho_*}{\rho_0} + \frac{\rho'}{\rho_0} \right ) \, , \\
                                           & = \partial_z \phi_{\rm{hyd}} - g \left ( 1 + \frac{\rho_*}{\rho_0} \right ) \, .
\end{align}
```
Under this pressure decomposition the pressure gradient that appears in the momentum equations becomes
```math
   \boldsymbol{\nabla} \phi \mapsto \boldsymbol{\nabla} \phi_{\rm{non}} + \boldsymbol{\nabla}_h \phi_{\rm{hyd}}\, .
```
Mathematically, the non-hydrostatic potential ``\phi_{\rm{non}}`` enforces the incompressibility constraint.
