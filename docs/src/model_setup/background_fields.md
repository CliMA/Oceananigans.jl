# Background fields

`BackgroundField`s are velocity and tracer fields around which the resolved
velocity and tracer fields evolve. In `Oceananigans`, only the _advective_ terms
associated with the interaction between background and resolved fields are included.
For example, tracer advection is described by

```math
\boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{v} c \right ) \, ,
```

where ``\boldsymbol{v}`` is the resolved velocity field and ``c`` is the resolved
tracer field corresponding to `model.tracers.c`. 

When a background field ``C`` is provided, the tracer advection term becomes

```math
\boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{v} c \right ) 
    + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{v} C \right ) \, .
```

When both a background field velocity field ``\boldsymbol{U}`` and a background tracer field ``C``
are provided, then the tracer advection term becomes

```math
\boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{v} c \right ) 
    + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{v} C \right )
    + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{U} c \right ) \, .
```

Notice that the term ``\boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \boldsymbol{U} C \right )`` 
is neglected: only the terms describing the advection of resolved tracer by the background 
velocity field and the advection of background tracer by the resolved velocity field are included.
An analgous statement holds for the advection of background momentum by the resolved
velocity field.
Other possible terms associated with the Coriolis force, buoyancy, turbulence closures,
and surface waves acting on background fields are neglected.

## Specifying background fields

`BackgroundField`s are defined by functions of ``(x, y, z, t)`` and optional parameters. A 
simple example is

```jldoctest
using Oceananigans

U(x, y, z, t) = 0.2 * z

grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))

model = NonhydrostaticModel(grid = grid, background_fields = (u=U,))

model.background_fields.velocities.u

# output
FunctionField located at (Face, Center, Center)
├── func: U
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
├── clock: Clock(time=0 seconds, iteration=0)
└── parameters: nothing
```

`BackgroundField`s are specified by passing them to the kwarg `background_fields`
in the `NonhydrostaticModel` constructor. The kwarg `background_fields` expects
a `NamedTuple` of fields, which are internally sorted into `velocities` and `tracers`,
wrapped in `FunctionField`s, and assigned their appropriate locations.

`BackgroundField`s with parameters require using the `BackgroundField` wrapper:

```jldoctest moar_background
using Oceananigans

parameters = (α=3.14, N=1.0, f=0.1)

## Background fields are defined via function of x, y, z, t, and optional parameters
U(x, y, z, t, α) = α * z
B(x, y, z, t, p) = - p.α * p.f * y + p.N^2 * z 

U_field = BackgroundField(U, parameters=parameters.α)
B_field = BackgroundField(B, parameters=parameters)

# output
BackgroundField{typeof(B), NamedTuple{(:α, :N, :f), Tuple{Float64, Float64, Float64}}}
├── func: B
└── parameters: (α = 3.14, N = 1.0, f = 0.1)
```

When inserted into `NonhydrostaticModel`, we get out

```jldoctest moar_background
grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))

model = NonhydrostaticModel(grid = grid, background_fields = (u=U_field, b=B_field),
                            tracers=:b, buoyancy=BuoyancyTracer())

model.background_fields.tracers.b

# output
FunctionField located at (Center, Center, Center)
├── func: B
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
├── clock: Clock(time=0 seconds, iteration=0)
└── parameters: (α = 3.14, N = 1.0, f = 0.1)
```
