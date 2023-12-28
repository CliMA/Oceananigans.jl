import Adapt

using Oceananigans.Utils: prettysummary

"""
    struct ParticleDiscreteForcing{P, F}

Wrapper for "discrete form" forcing functions for particles with optional `parameters`.
"""
struct ParticleDiscreteForcing{P, F}
    func :: F
    parameters :: P
end

"""
    ParticleDiscreteForcing(func; parameters=nothing)

Construct a "discrete form" forcing function for Lagrangian particles with optional parameters.
The forcing function is applied for particle `p`.

When `parameters` are not specified, `func` must be callable with the signature

```
func(i, j, k, particles, p, grid, clock, Δt, model_fields)
```

where `grid` is `model.grid`, `clock.time` is the current simulation time and
`clock.iteration` is the current model iteration, and `model_fields` is a
`NamedTuple` with `u, v, w`, the fields in `model.tracers`, and any auxillary fields.

*Note* that the index `end` does *not* access the final physical grid point of
a model field in any direction. The final grid point must be explicitly specified, as
in `model_fields.u[i, j, grid.Nz]`.

When `parameters` _is_ specified, `func` must be callable with the signature.

```
func(i, j, k, particles, p, grid, clock, Δt, model_fields, parameters)
```
    
Above, `parameters` is, in principle, arbitrary. Note, however, that GPU compilation
can place constraints on `typeof(parameters)`.
"""
ParticleDiscreteForcing(func; parameters=nothing) = ParticleDiscreteForcing(func, parameters)

@inline no_discrete_forcing(x, y, z, fluid_velocity, args...) = fluid_velocity
ParticleDiscreteForcing() = ParticleDiscreteForcing(no_discrete_forcing)

@inline function (forcing::ParticleDiscreteForcing{P, F})(x, y, z, fluid_velocity, particles, p, grid, clock, Δt, model_fields) where {P, F<:Function}
    parameters = forcing.parameters
    return forcing.func(x, y, z, fluid_velocity, particles, p, grid, clock, Δt, model_fields, parameters)
end

@inline (forcing::ParticleDiscreteForcing{<:Nothing, F})(x, y, z, fluid_velocity, particles, p, grid, clock, Δt, model_fields) where F<:Function =
    forcing.func(x, y, z, fluid_velocity, particles, p, grid, clock, Δt, model_fields)

"""Show the innards of a `ParticleDiscreteForcing` in the REPL."""
Base.show(io::IO, forcing::ParticleDiscreteForcing{P}) where P =
    print(io, "ParticleDiscreteForcing{$P}", "\n",
        "├── func: $(prettysummary(forcing.func))", "\n",
        "└── parameters: $(forcing.parameters)")

Adapt.adapt_structure(to, forcing::ParticleDiscreteForcing) =
    ParticleDiscreteForcing(Adapt.adapt(to, forcing.func),
                    Adapt.adapt(to, forcing.parameters))
