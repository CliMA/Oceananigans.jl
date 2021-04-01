# Pressure decomposition

In the numerical implementation of the momentum equations, the kinematic potential ``\phi`` 
is split into "hydrostatic anomaly" and "non-hydrostatic" parts via
```math
    \tag{eq:pressure}
    \phi(\boldsymbol{x}, t) = \phi_{\rm{hyd}}(\boldsymbol{x}, t) + \phi_{\rm{non}}(\boldsymbol{x}, t) \, .
```
The anomalous hydrostatic component of the kinematic potential is defined by 
```math
    \tag{eq:hydrostaticpressure}
    \partial_z \phi_{\rm{hyd}} \equiv -b \, ,
```
such that the sum of the kinematic potential and buoyancy perturbation becomes
```math
    -\boldsymbol{\nabla} \phi + b \boldsymbol{\hat z} = 
        - \boldsymbol{\nabla} \phi_{\rm{non}}
        - \big ( \underbrace{\partial_x \boldsymbol{\hat x} + \partial_y \boldsymbol{\hat y} }_{\equiv \boldsymbol{\nabla}_{\! h}} \big ) \phi_{\rm{hyd}} \, .
```
The hydrostatic pressure anomaly is so named because the "total" hydrostatic pressure 
contains additional components:
```math
\begin{aligned}
\partial_z \phi_{\text{total hydrostatic}} &= - g \left ( 1 + \tfrac{\rho_*}{\rho_0} + \tfrac{\rho'}{\rho_0} \right ) \, , \\
                                           &= \partial_z \phi_{\rm{hyd}} - g \left ( 1 + \tfrac{\rho_*}{\rho_0} \right ) \, .
\end{aligned}
```
Under this pressure decomposition the momentum equation becomes
```math
   \partial_t \boldsymbol{u} + \left ( \boldsymbol{u} \boldsymbol{\cdot} \boldsymbol{\nabla} \right ) \boldsymbol{u} + \boldsymbol{f} \times \boldsymbol{u} = 
    - \boldsymbol{\nabla} \phi_{\rm{non}} - \boldsymbol{\nabla}_h \phi_{\rm{hyd}} - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\tau} + \boldsymbol{F_u} \, .
```
Mathematically, the non-hydrostatic potential ``\phi_{\rm{non}}`` enforces the incompressibility constraint.
