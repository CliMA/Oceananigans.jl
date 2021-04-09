# Buoyancy model and equations of state

The buoyancy model determines the relationship between tracers and the buoyancy ``b`` in the momentum equation.

## Buoyancy tracer

The simplest buoyancy model uses buoyancy ``b`` itself as a tracer: ``b`` obeys the tracer
conservation equation and is used directly in the momentum equations in the momentum equation.

## Seawater buoyancy

For seawater buoyancy is, in general, modeled as a function of conservative temperature
``\theta``, absolute salinity ``S``, and depth below the ocean surface ``d`` via
```math
    \begin{equation}
    b = - \frac{g}{\rho_0} \rho' \left (\theta, S, d \right ) \, ,
    \label{eq:seawater-buoyancy}
    \end{equation}
```
where ``g`` is gravitational acceleration, ``\rho_0`` is the reference density.
The function ``\rho'(\theta, S, d)`` in the seawater buoyancy relationship that links conservative temperature,
salinity, and depth to the density perturbation is called the *equation of state*.
Both ``\theta`` and ``S`` obey the tracer conservation equation.

### Linear equation of state

Buoyancy is determined from a linear equation of state via
```math
    b = g \left ( \alpha_\theta \theta - \beta_S S \right ) \, ,
```
where ``g`` is gravitational acceleration, ``\alpha_\theta`` is the thermal expansion coefficient,
and ``\beta_S`` is the haline contraction coefficient.

### Nonlinear equation of state

Buoyancy is determined by the simplified equations of state introduced by [Roquet15TEOS](@cite).
