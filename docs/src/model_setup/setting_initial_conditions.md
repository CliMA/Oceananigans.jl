# Setting initial conditions
Initial conditions are imposed after model construction. This can be easily done using the the `set!` function, which
allows the setting of initial conditions using constant values, arrays, or functions.

```@example
set!(model.velocities.u, 0.1)
```

```@example
∂T∂z = 0.01
ϵ(σ) = σ * randn()
T₀(x, y, z) = ∂T∂z * z + ϵ(1e-8)
set!(model.tracers.T, T₀)
```
