# Buoyancy models and equations of state

The buoyancy option selects how buoyancy is treated in `NonhydrostaticModel`s and
`HydrostaticFreeSurfaceModel`s (`ShallowWaterModel`s do not have that option given the physics of
the model). There are currently three alternatives:

1. No buoyancy (and no gravity).
2. Evolve buoyancy as a tracer.
3. _Seawater buoyancy_: evolve temperature ``T`` and salinity ``S`` as tracers with a value for the gravitational
   acceleration ``g`` and an equation of state of your choosing.

## No buoyancy

To turn off buoyancy (and gravity) you can simply pass `buoyancy = nothing` to the model
constructor. For example to create a `NonhydrostaticModel`:


```@meta
DocTestSetup = quote
    using Oceananigans
end
```


```jldoctest buoyancy
julia> grid = RectilinearGrid(size=(64, 64, 64), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid=grid, buoyancy=nothing)
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing
```

`buoyancy=nothing` is the default option for`NonhydrostaticModel`, so ommitting `buoyancy`
from the `NonhydrostaticModel` constructor yields an identical result:

```jldoctest buoyancy
julia> model = NonhydrostaticModel(grid=grid)
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing
```

To create a `HydrostaticFreeSurfaceModel` without a buoyancy term we explicitly
specify `buoyancy=nothing` flag. The default tracers `T` and `S` for `HydrostaticFreeSurfaceModel`
may be eliminated when `buoyancy=nothing` by specifying `tracers=()`:

```jldoctest buoyancy
julia> model = HydrostaticFreeSurfaceModel(grid=grid, buoyancy=nothing, tracers=())
HydrostaticFreeSurfaceModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
├── free surface: ExplicitFreeSurface with gravitational acceleration 9.80665 m s⁻²
└── coriolis: Nothing
```

## Buoyancy as a tracer

Both `NonhydrostaticModel` and `HydrostaticFreeSurfaceModel` support evolving
a buoyancy tracer by including `:b` in `tracers` and specifying  `buoyancy = BuoyancyTracer()`:

```jldoctest buoyancy
julia> model = NonhydrostaticModel(grid=grid, buoyancy=BuoyancyTracer(), tracers=:b)
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── tracers: (:b,)
├── closure: Nothing
├── buoyancy: Buoyancy{BuoyancyTracer, Oceananigans.Grids.ZDirection}
└── coriolis: Nothing
```

We follow the same pattern to create a `HydrostaticFreeSurfaceModel` with buoyancy as a tracer:

```jldoctest buoyancy
julia> model = HydrostaticFreeSurfaceModel(grid=grid, buoyancy=BuoyancyTracer(), tracers=:b)
HydrostaticFreeSurfaceModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── tracers: (:b,)
├── closure: Nothing
├── buoyancy: Buoyancy{BuoyancyTracer, Oceananigans.Grids.ZDirection}
├── free surface: ExplicitFreeSurface with gravitational acceleration 9.80665 m s⁻²
└── coriolis: Nothing
```

## Seawater buoyancy

`NonhydrostaticModel` and `HydrostaticFreeSurfaceModel` support modeling the buoyancy of seawater
as a function of gravitational acceleration, conservative temperature ``T`` and absolute salinity ``S``.
The relationship between ``T``, ``S``, the geopotential height, and the density perturbation from
a reference value is called the `equation_of_state`.
Specifying `buoyancy = SeawaterBuoyancy()` (which uses a linear equation of state and
[Earth standard](https://en.wikipedia.org/wiki/Standard_gravity)
`gravitational_acceleration = 9.80665 \, \text{m}\,\text{s}^{-2}` by default)
requires the tracers `:T` and `:S`:

```jldoctest buoyancy
julia> model = NonhydrostaticModel(grid=grid, buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── tracers: (:T, :S)
├── closure: Nothing
├── buoyancy: Buoyancy{SeawaterBuoyancy{Float64, LinearEquationOfState{Float64}, Nothing, Nothing}, Oceananigans.Grids.ZDirection}
└── coriolis: Nothing
```

With `HydrostaticFreeSurfaceModel`,

```jldoctest buoyancy
julia> model = HydrostaticFreeSurfaceModel(grid=grid, buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
HydrostaticFreeSurfaceModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── tracers: (:T, :S)
├── closure: Nothing
├── buoyancy: Buoyancy{SeawaterBuoyancy{Float64, LinearEquationOfState{Float64}, Nothing, Nothing}, Oceananigans.Grids.ZDirection}
├── free surface: ExplicitFreeSurface with gravitational acceleration 9.80665 m s⁻²
└── coriolis: Nothing
```

is identical to the default,

```jldoctest buoyancy
julia> model = HydrostaticFreeSurfaceModel(grid=grid)
HydrostaticFreeSurfaceModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── tracers: (:T, :S)
├── closure: Nothing
├── buoyancy: Buoyancy{SeawaterBuoyancy{Float64, LinearEquationOfState{Float64}, Nothing, Nothing}, Oceananigans.Grids.ZDirection}
├── free surface: ExplicitFreeSurface with gravitational acceleration 9.80665 m s⁻²
└── coriolis: Nothing
```

To model flows near the surface of Europa where `gravitational_acceleration = 1.3 \, \text{m}\,\text{s}^{-2}`,
we might alternatively specify

```jldoctest buoyancy
julia> buoyancy = SeawaterBuoyancy(gravitational_acceleration=1.3)
SeawaterBuoyancy{Float64}: g = 1.3
└── equation of state: LinearEquationOfState{Float64}: α = 1.67e-04, β = 7.80e-04

julia> model = NonhydrostaticModel(grid=grid, buoyancy=buoyancy, tracers=(:T, :S))
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── tracers: (:T, :S)
├── closure: Nothing
├── buoyancy: Buoyancy{SeawaterBuoyancy{Float64, LinearEquationOfState{Float64}, Nothing, Nothing}, Oceananigans.Grids.ZDirection}
└── coriolis: Nothing
```

for example.

### Linear equation of state

To specify the thermal expansion and haline contraction coefficients
``\alpha = 2 \times 10^{-3} \; \text{K}^{-1}`` and ``\beta = 5 \times 10^{-4} \text{psu}^{-1}``,

```jldoctest
julia> buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-3, β=5e-4))
SeawaterBuoyancy{Float64}: g = 9.80665
└── equation of state: LinearEquationOfState{Float64}: α = 2.00e-03, β = 5.00e-04
```

### Idealized nonlinear equations of state

Instead of a linear equation of state, five idealized (second-order) nonlinear equation of state as described by
[Roquet15Idealized](@cite) may be used. These equations of state are provided via the
[SeawaterPolynomials.jl](https://github.com/CliMA/SeawaterPolynomials.jl) package.

```jldoctest buoyancy
julia> using SeawaterPolynomials.SecondOrderSeawaterPolynomials

julia> eos = RoquetSeawaterPolynomial(:Freezing)
SecondOrderSeawaterPolynomial{Float64}(0.7718, -0.0491, 0.0, -2.5681e-5, 0.0, -0.005027, 0.0)

julia> buoyancy = SeawaterBuoyancy(equation_of_state=eos)
SeawaterBuoyancy{Float64}: g = 9.80665
└── equation of state: SeawaterPolynomials.SecondOrderSeawaterPolynomials.SecondOrderSeawaterPolynomial{Float64}(0.7718, -0.0491, 0.0, -2.5681e-5, 0.0, -0.005027, 0.0)
```

### TEOS-10 equation of state

A high-accuracy 55-term polynomial approximation to the TEOS-10 equation of state suitable for use in
Boussinesq models as described by [Roquet15TEOS](@cite) is implemented in the
[SeawaterPolynomials.jl](https://github.com/CliMA/SeawaterPolynomials.jl) package and may be used.

```jldoctest buoyancy
julia> using SeawaterPolynomials.TEOS10

julia> eos = TEOS10EquationOfState()
SeawaterPolynomials.BoussinesqEquationOfState{TEOS10SeawaterPolynomial{Float64}, Int64}(TEOS10SeawaterPolynomial{Float64}(), 1020)
```

## The direction of gravitational acceleration

To simulate gravitational accelerations that don't align with the vertical (`z`) coordinate,
we wrap the buoyancy model in
`Buoyancy()` function call, which takes the keyword arguments `model` and `vertical_unit_vector`,

```jldoctest buoyancy
julia> θ = 45; # degrees

julia> g̃ = (0, sind(θ), cosd(θ));

julia> model = NonhydrostaticModel(grid=grid, 
                                   buoyancy=Buoyancy(model=BuoyancyTracer(), vertical_unit_vector=g̃), 
                                   tracers=:b)
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── tracers: (:b,)
├── closure: Nothing
├── buoyancy: Buoyancy{BuoyancyTracer, Tuple{Int64, Float64, Float64}}
└── coriolis: Nothing
```

