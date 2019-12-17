# Setting initial conditions
Initial conditions are imposed after model construction. This can be easily done using the the `set!` function, which
allows the setting of initial conditions using constant values, arrays, or functions.

```@example
set!(model, u=0.1, v=1.5)
```

```@example
∂T∂z = 0.01
ϵ(σ) = σ * randn()
T₀(x, y, z) = ∂T∂z * z + ϵ(1e-8)
set!(model, T=T₀)
```

!!! tip "Divergence-free velocity fields"
    Note that as part of the time-stepping algorithm, the velocity field is made
    divergence-free at every time step. So if a model is not initialized with a
    divergence-free velocity field, it may change on the first time step. As a result
    tracers may not be conserved up to machine precision at the first time step.
