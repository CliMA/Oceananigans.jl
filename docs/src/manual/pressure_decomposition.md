# Pressure decomposition

In the numerical implementation of the momentum equations, the kinematic potential $\phi$ 
is split into "hydrostatic anomaly" and "non-hydrostatic" parts via
```math
    \tag{eq:pressure}
    \phi(x, y, z) = \phi_{\rm{hyd}}(x, y, z) + \phi_{\rm{non}}(x, y, z)
```
The anomalous hydrostatic component of the kinematic potential is defined by 
```math
    \tag{eq:hydrostaticpressure}
    \partial_z \phi_{\rm{hyd}} \equiv -b
```
such that the sum of the kinematic potential and buoyancy perturbation becomes
```math
    -\bm{\nabla} \phi + b \bm{\hat z} = 
        - \bm{\nabla} \phi_n 
        - \big ( \underbrace{\partial_x \bm{\hat x} + \partial_y \bm{\hat y} }_{\equiv \bm{\nabla}_{\! h}} \big ) \phi_{\rm{hyd}} \, .
```
The hydrostatic pressure anomaly is so named because the "total" hydrostatic pressure contains additional components such that
```math
\partial_z \phi_{\text{total hydrostatic}} = - g \left ( 1 + \tfrac{\rho_*}{\rho_0} + \tfrac{\rho'}{\rho_0} \right ) = \partial_z \phi_{\rm{hyd}} - g \left ( 1 + \tfrac{\rho_*}{\rho_0} \right ) \, .
```
Under this pressure decomposition the momentum equation becomes
```math
   \partial_t \bm{u} + \left ( \bm{u} \bm{\cdot} \bm{\nabla} \right ) \bm{u} + \bm{f} \times \bm{u} = - \bm{\nabla} \phi_{\rm{non}} - \bm{\nabla}_h \phi_{\rm{hyd}} - \bm{\nabla} \bm{\cdot} \bm{\tau} + \bm{F_u} \, .
```
Mathematically, the non-hydrostatic potential $\phi_{\rm{non}}$ enforces the incompressibility constraint.
