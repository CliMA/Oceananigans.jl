"""
    Forcing(func; parameters=nothing, field_dependencies=(), discrete_form=false)

Returns a forcing function added to the tendency of an Oceananigans model field.

If `discrete_form=false` (the default), and neither `parameters` nor `field_dependencies`
are provided, then `func` must be callable with the signature

    `func(x, y, z, t)`

where `x, y, z` are the east-west, north-south, and vertical spatial coordinates, and `t` is time.
Note that this form is also default in the constructor for `NonhydrostaticModel`, so that `Forcing` is
not needed.

If `discrete_form=false` (the default), and `field_dependencies` are provided,
the signature of `func` must include them. For example, if `field_dependencies=(:u, :S)`
(and `parameters` are _not_ provided), then `func` must be callable with the signature

    `func(x, y, z, t, u, S)`

where `u` is assumed to be the `u`-velocity component, and `S` is a tracer. Note that any field
which does not have the name `u`, `v`, or `w` is assumed to be a tracer and must be present
in `model.tracers`.

If `discrete_form=false` (the default) and `parameters` are provided, then the _last_ argument
to `func` must be `parameters`. For example, if `func` has no `field_dependencies` but does
depend on `parameters`, then it must be callable with the signature

    `func(x, y, z, t, parameters)`

The object `parameters` is arbitrary in principle, however GPU compilation can place
constraints on `typeof(parameters)`.

With `field_dependencies=(:u, :v, :w, :c)` and `parameters`, then `func` must be
callable with the signature

    `func(x, y, z, t, u, v, w, c, parameters)`

If `discrete_form=true` then `func` must be callable with the "discrete form"

    `func(i, j, k, grid, clock, model_fields)`

where `i, j, k` is the grid point at which the forcing is applied, `grid` is `model.grid`,
`clock.time` is the current simulation time and `clock.iteration` is the current model iteration,
and `model_fields` is a `NamedTuple` with `u, v, w`, the fields in `model.tracers`,
and the fields in `model.diffusivities`, each of which is an `OffsetArray`s (or `NamedTuple`s
of `OffsetArray`s depending on the turbulence closure) of field data.

When `discrete_form=true` and `parameters` _is_ specified, `func` must be callable with the signature

    `func(i, j, k, grid, clock, model_fields, parameters)`

Examples
========

```jldoctest forcing
using Oceananigans

# Parameterized forcing
parameterized_func(x, y, z, t, p) = p.μ * exp(z / p.λ) * cos(p.ω * t)

v_forcing = Forcing(parameterized_func, parameters = (μ=42, λ=0.1, ω=π))

# output
ContinuousForcing{NamedTuple{(:μ, :λ, :ω), Tuple{Int64, Float64, Irrational{:π}}}}
├── func: parameterized_func
├── parameters: (μ = 42, λ = 0.1, ω = π)
└── field dependencies: ()
```

Note that because forcing locations are regularized within the
`NonhydrostaticModel` constructor:

```jldoctest forcing
grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=grid, forcing=(v=v_forcing,))

model.forcing.v

# output
ContinuousForcing{NamedTuple{(:μ, :λ, :ω), Tuple{Int64, Float64, Irrational{:π}}}} at (Center, Face, Center)
├── func: parameterized_func
├── parameters: (μ = 42, λ = 0.1, ω = π)
└── field dependencies: ()
```

After passing through the constructor for `NonhydrostaticModel`, the `v`-forcing location
information is available and set to `Center, Face, Center`.

```jldoctest forcing
# Field-dependent forcing
growth_in_sunlight(x, y, z, t, P) = exp(z) * P

plankton_forcing = Forcing(growth_in_sunlight, field_dependencies=:P)

# output
ContinuousForcing{Nothing}
├── func: growth_in_sunlight
├── parameters: nothing
└── field dependencies: (:P,)
```

```jldoctest forcing
# Parameterized, field-dependent forcing
tracer_relaxation(x, y, z, t, c, p) = p.μ * exp((z + p.H) / p.λ) * (p.dCdz * z - c) 

c_forcing = Forcing(tracer_relaxation,
                    field_dependencies = :c,
                            parameters = (μ=1/60, λ=10, H=1000, dCdz=1))

# output
ContinuousForcing{NamedTuple{(:μ, :λ, :H, :dCdz), Tuple{Float64, Int64, Int64, Int64}}}
├── func: tracer_relaxation
├── parameters: (μ = 0.016666666666666666, λ = 10, H = 1000, dCdz = 1)
└── field dependencies: (:c,)
```

```jldoctest forcing
# Unparameterized discrete-form forcing function
filtered_relaxation(i, j, k, grid, clock, model_fields) =
    @inbounds - (model_fields.c[i-1, j, k] + model_fields.c[i, j, k] + model_fields.c[i+1, j, k]) / 3

filtered_forcing = Forcing(filtered_relaxation, discrete_form=true)

# output
DiscreteForcing{Nothing}
├── func: filtered_relaxation
└── parameters: nothing
```

```jldoctest forcing
# Discrete-form forcing function with parameters
masked_damping(i, j, k, grid, clock, model_fields, parameters) = 
    @inbounds - parameters.μ * exp(grid.zC[k] / parameters.λ) * model_fields.u[i, j, k]

masked_damping_forcing = Forcing(masked_damping, parameters=(μ=42, λ=π), discrete_form=true)

# output
DiscreteForcing{NamedTuple{(:μ, :λ), Tuple{Int64, Irrational{:π}}}}
├── func: masked_damping
└── parameters: (μ = 42, λ = π)
```
"""
function Forcing(func; parameters=nothing, field_dependencies=(), discrete_form=false)
    if discrete_form
        return DiscreteForcing(func; parameters=parameters)
    else
        return ContinuousForcing(func; parameters=parameters, field_dependencies=field_dependencies)
    end
end
