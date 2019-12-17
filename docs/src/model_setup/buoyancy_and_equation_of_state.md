# Buoyancy and equation of state
The buoyancy option selects how buoyancy is treated. There are currently three options:
1. No buoyancy (and no gravity).
2. Evolve buoyancy as a tracer.
3. _Seawater buoyancy_: evolve temperature $T$ and salinity $S$ as tracers with a value for the gravitational
   acceleration $g$ and an appropriate equation of state.

## No buoyancy
To turn off buoyancy (and gravity) simply pass
```
buoyancy = nothing
```
to the `Model` constructor. In this case, you will probably also want to explicitly specify which tracers to evolve.
In particular, you probably will not want to evolve temperature and salinity, which are included by default. To specify
no tracers, also pass
```
tracers = ()
```
to the `Model` constructor.

## Buoyancy as a tracer
To directly evolve buoyancy as a tracer simply pass
```@example
using Oceananigans # hide
buoyancy = BuoyancyTracer()
```
to the `Model` constructor. Buoyancy `:b` must be included as a tracer, for example, by also passing
```
tracers = (:b)
```

## Seawater buoyancy
To evolve temperature $T$ and salinity $S$ and diagnose the buoyancy, you can pass
```@example
using Oceananigans # hide
buoyancy = SeawaterBuoyancy()
```
which is also the default. Without any options specified, a value of $g = 9.80665 \; \text{m/s}^2$ is used for the
gravitational acceleration (corresponding to [standard gravity](https://en.wikipedia.org/wiki/Standard_gravity)) along
with a linear equation of state with thermal expansion and haline contraction coefficients suitable for water.

If, for example, you wanted to simulate fluids on another planet such as Europa where $g = 1.3 \; \text{m/s}^2$, then
use
```@example
using Oceananigans # hide
buoyancy = SeawaterBuoyancy(gravitational_acceleration=1.3)
```

When using `SeawaterBuoyancy` temperature `:T` and salinity `:S` tracers must be specified
```
tracers = (:T, :S)
```

### Linear equation of state
To use non-default thermal expansion and haline contraction coefficients, say
$\alpha = 2 \times 10^{-3} \; \text{K}^{-1}$ and $\beta = 5 \times 10^{-4} \text{ppt}^{-1}$ corresponding to some other
fluid, then use

```@example
using Oceananigans # hide
buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState(α=1.67e-4, β=7.80e-4))
```

### Idealized nonlinear equation of state
Instead of a linear equation of state, five idealized nonlinear equation of state as described by Roquet et al. (2015)
may be specified. See [`RoquetIdealizedNonlinearEquationOfState`](@ref RoquetIdealizedNonlinearEquationOfState).
