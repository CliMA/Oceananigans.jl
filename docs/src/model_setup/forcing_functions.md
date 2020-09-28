# Forcing functions

"Forcings" are user-defined terms that are appended to right-hand side of
the momentum or tracer evolution equations. In `Oceananigans`, momentum
and tracer forcings are defined via julia functions. `Oceananigans` includes
an interface for implementing forcing functions that depend on spatial coordinates,
time, other model fields, and external parameters.

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

Forcings are added to `Oceananigans` models by passing a `NamedTuple` of functions
or forcing objects to the constructor for `IncompressibleModel`.
By default, momentum and tracer forcing functions are assumed to be functions of
`x, y, z, t`. A basic example is

```jldoctest
u_forcing(x, y, z, t) = exp(z) * cos(x) * sin(t)

grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = IncompressibleModel(grid=grid, forcing=(u=u_forcing,))

# output
IncompressibleModel{CPU, Float64}(time = 0.000 s, iteration = 0)
├── grid: RegularCartesianGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
├── tracers: (:T, :S)
├── closure: IsotropicDiffusivity{Float64,NamedTuple{(:T, :S),Tuple{Float64,Float64}}}
├── buoyancy: SeawaterBuoyancy{Float64,LinearEquationOfState{Float64},Nothing,Nothing}
└── coriolis: Nothing
```

More general forcing functions are built via the `Forcing` constructor.
`Oceananigans` also provides a convenience type called `Relaxation` that
specifies "relaxation", or damping terms that restore a field to a
target distribution outside of a masked region of space.

## The `Forcing` constructor

The `Forcing` constructor provides an interface for specifying forcing functions that

1. Depend on external parameters;
2. Depend on model fields at the `x, y, z` location that forcing is applied; or
3. The discrete grid and global model field data.

### Forcing functions with external parameters

Here's an example of a `forcing_func`tion that depends on a `NamedTuple` of parameters, `p`:

```jldoctest
# Forcing that depends on a scalar parameter `s`
u_forcing_func(x, y, z, t, s) = s * z

u_forcing = Forcing(u_forcing_func, parameters=1)

# Forcing that depends on a `NamedTuple` of parameters `p`
T_forcing_func(x, y, z, t, p) = - p.μ * exp(z / p.λ) * cos(p.k * x) * sin(p.ω * t)

T_forcing = Forcing(T_forcing_func, parameters=(μ=1, λ=0.5, k=2π, ω=4π))

grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = IncompressibleModel(grid=grid, forcing=(u=u_forcing, T=T_forcing))

# output
IncompressibleModel{CPU, Float64}(time = 0.000 s, iteration = 0)
├── grid: RegularCartesianGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
├── tracers: (:T, :S)
├── closure: IsotropicDiffusivity{Float64,NamedTuple{(:T, :S),Tuple{Float64,Float64}}}
├── buoyancy: SeawaterBuoyancy{Float64,LinearEquationOfState{Float64},Nothing,Nothing}
└── coriolis: Nothing
```

In the above example, the objects passed to the `parameters` keyword in the construction of
`u_forcing` and `T_forcing` --- an integer for `u_forcing`, and a `NamedTuple` of parameters
for `T_forcing` --- are passed on to `u_forcing_func` and `T_forcing_func` when they are
called during time-stepping. The object passed to `parameters` is in principle arbitrary.
However, if using the GPU, then `typeof(parameters)` may be restricted by the requirements
of GPU-compiliability.

### Forcing functions that depend on model fields

Forcing functions may depend on other model fields at the location at which forcing is applied.
Here's a somewhat non-sensical example:

```jldoctest
# Forcing that depends on the velocity fields `u`, `v`, and `w`
w_forcing_func(x, y, z, t, u, v, w) = - (u^2 + v^2 + w^2) / 2

w_forcing = Forcing(w_forcing_func, field_dependencies=(:u, :v, :w))

# Forcing that depends on temperature `T` and a scalar parameter
S_forcing_func(x, y, z, t, S, μ) = - μ * S

S_forcing = Forcing(S_forcing_func, parameters=1/60, field_dependencies=:S)

grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = IncompressibleModel(grid=grid, forcing=(w=w_forcing, S=S_forcing))

# output
IncompressibleModel{CPU, Float64}(time = 0.000 s, iteration = 0)
├── grid: RegularCartesianGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
├── tracers: (:T, :S)
├── closure: IsotropicDiffusivity{Float64,NamedTuple{(:T, :S),Tuple{Float64,Float64}}}
├── buoyancy: SeawaterBuoyancy{Float64,LinearEquationOfState{Float64},Nothing,Nothing}
└── coriolis: Nothing
```

The `field_dependencies` arguments follow `x, y, z, t` in the forcing `func`tion in
the order they are specified in `Forcing`.
If both `field_dependencies` and `parameters` are specified, then the `field_dependencies`
arguments follow `x, y, z, t`, and `parameters` follow `field_dependencies`.

### "Discrete form" forcing functions

"Discrete form" forcing functions are either called with the signature

```julia
f(i, j, k, grid, clock, model_fields)
```

or, when specified with parameters, are called with

```julia
f(i, j, k, grid, clock, model_fields, parameters)
```

Discrete form forcing functions give users access to the entirety of model field
data through the argument `model_fields`. The object `model_fields` is a `NamedTuple`
whose properties include the velocity fields `model_fields.u`, `model_fields.v`,
`model_fields.w`, all fields in `model.tracers`, and the fields in `model.diffusivities`.
Note that in the special case that a _tuple_ of turbulence closures is provided,
the `diffusivities` associated with `closure[i]` is accessible via
`model_fields.diffusivities[i]`.

Using discrete forcing functions often requires an understanding of the
staggered arrangement of variables employed in Oceananigans.
Here's a slightly non-sensical example in which the vertical derivative of a buoyancy
tracer is used as a time-scale for damping the u-velocity field:

```jldoctest
# A damping term that depends on a "local average" of `u`:
local_average(i, j, k, grid, c) = @inbounds (6 * c[i, j, k] + c[i-1, j, k] + c[i+1, j, k] +
                                                              c[i, j-1, k] + c[i, j+1, k] +
                                                              c[i, j, k-1] + c[i, j, k+1])

b_forcing_func(i, j, k, grid, clock, model_fields) = - local_average(i, j, k, grid, model_fields.b)

b_forcing = Forcing(b_forcing_func, discrete_form=true)

# A term that damps the local velocity field in the presence of stratification
using Oceananigans.Operators: ∂zᵃᵃᶠ, ℑxzᶠᵃᶜ

function u_forcing_func(i, j, k, grid, clock, model_fields, ε)
    # The vertical derivative of buoyancy, interpolated to the u-velocity location:
    N² = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, model_fields.b)

    # Set to zero in unstable stratification where N² < 0:
    N² = max(N², zero(typeof(N²)))

    return @inbounds - ε / sqrt(N²) * model_fields.u[i, j, k]
end

u_forcing = Forcing(u_forcing_func, discrete_form=true, parameters=1e-3)

grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1))
model = IncompressibleModel(grid=grid, tracers=:b, buoyancy=BuoyancyTracer(), forcing=(u=u_forcing, b=b_forcing))

# output
IncompressibleModel{CPU, Float64}(time = 0.000 s, iteration = 0)
├── grid: RegularCartesianGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
├── tracers: (:b,)
├── closure: IsotropicDiffusivity{Float64,NamedTuple{(:b,),Tuple{Float64}}}
├── buoyancy: BuoyancyTracer
└── coriolis: Nothing
```

The annotation `@inbounds` is crucial for performance when accessing array indices
of the fields in `model_fields`.
