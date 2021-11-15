# Buoyancy models and equations of state

The buoyancy option selects how buoyancy is treated. There are currently three options:

1. No buoyancy (and no gravity).
2. Evolve buoyancy as a tracer.
3. _Seawater buoyancy_: evolve temperature ``T`` and salinity ``S`` as tracers with a value for the gravitational
   acceleration ``g`` and an equation of state of your choosing. This is the default setting.

## No buoyancy

To turn off buoyancy (and gravity) simply pass `buoyancy = nothing` to the model constructor.

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest buoyancy
julia> grid = RectilinearGrid(size=(64, 64, 64), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid=grid, buoyancy=nothing, tracers=(:T, :S))
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: (:T, :S)
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing
```

In this case, you might want to explicitly specify which tracers to evolve. In particular, you may
not want to evolve temperature and salinity, which are included by default. To specify no tracers,
also pass`tracers = ()` to the model constructor.

```jldoctest buoyancy
julia> model = NonhydrostaticModel(grid=grid, buoyancy=nothing, tracers=())
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing
```

## Buoyancy as a tracer

To directly evolve buoyancy as a tracer simply pass `buoyancy = BuoyancyTracer()` to the model
constructor. BuoyancyModels `:b` must be included as a tracer, for example,

```jldoctest buoyancy
julia> model = NonhydrostaticModel(grid=grid, buoyancy=BuoyancyTracer(), tracers=(:b))
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: (:b,)
├── closure: Nothing
├── buoyancy: BuoyancyTracer
└── coriolis: Nothing
```

## Seawater buoyancy

To evolve temperature ``T`` and salinity ``S`` and diagnose the buoyancy, you can pass
`buoyancy = SeawaterBuoyancy()` which is the default.

```jldoctest buoyancy
julia> model = NonhydrostaticModel(grid=grid, buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: (:T, :S)
├── closure: Nothing
├── buoyancy: SeawaterBuoyancy{Float64, LinearEquationOfState{Float64}, Nothing, Nothing}
└── coriolis: Nothing
```

Without any options specified, a value of ``g = 9.80665 \, \text{m}\,\text{s}^{-2}`` is used for the gravitational
acceleration (corresponding to [standard gravity](https://en.wikipedia.org/wiki/Standard_gravity)) along
with a linear equation of state with thermal expansion and haline contraction coefficients suitable for seawater.

If, for example, you wanted to simulate fluids on another planet such as Europa where ``g = 1.3 \, \text{m}\,\text{s}^{-2}``,
then use

```jldoctest buoyancy
julia> buoyancy = SeawaterBuoyancy(gravitational_acceleration=1.3)
SeawaterBuoyancy{Float64}: g = 1.3
└── equation of state: LinearEquationOfState{Float64}: α = 1.67e-04, β = 7.80e-04

julia> model = NonhydrostaticModel(grid=grid, buoyancy=buoyancy, tracers=(:T, :S))
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: (:T, :S)
├── closure: Nothing
├── buoyancy: SeawaterBuoyancy{Float64, LinearEquationOfState{Float64}, Nothing, Nothing}
└── coriolis: Nothing
```

When using `SeawaterBuoyancy` temperature `:T` and salinity `:S` tracers must be specified. Explicitly this
can be accomplished by passing `tracers = (:T, :S)` to a model constructor.

### Linear equation of state

To use non-default thermal expansion and haline contraction coefficients, say
``\alpha = 2 \times 10^{-3} \; \text{K}^{-1}`` and ``\beta = 5 \times 10^{-4} \text{psu}^{-1}`` corresponding to some other
fluid, then use

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
