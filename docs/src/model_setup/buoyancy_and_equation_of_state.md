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
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing
```

`buoyancy=nothing` is the default option for`NonhydrostaticModel`, so you can achieve the same
result by simply creating a `NonhydrostaticModel` without explicitly passing the `buoyancy` flag:

```jldoctest buoyancy
julia> model = NonhydrostaticModel(grid=grid)
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing
```


In order to create a `HydrostaticFreeSurfaceModel` without any buoyancy treatment we need to pass
explicitly pass the `buoyancy=nothing` flag. Note that by default `HydrostaticFreeSurfaceModel`
advects temperature `T` and salinity `S` (which aren't necessary without a buoyancy treatment), so
it is often recommended to explicitly specify the tracers in this case as well:

```jldoctest buoyancy; filter = [r".*┌ Warning.*", r".*└ @ Oceananigans.*"]
julia> model = HydrostaticFreeSurfaceModel(grid=grid, buoyancy=nothing, tracers=())
┌ Warning: HydrostaticFreeSurfaceModel is experimental. Use with caution!
└ @ Oceananigans.Models.HydrostaticFreeSurfaceModels ~/builds/tartarus-3/clima/oceananigans/src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl:106
HydrostaticFreeSurfaceModel{CPU, Float64}(time = 0 seconds, iteration = 0) 
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing
```




## Buoyancy as a tracer

In order to directly evolve buoyancy as a tracer, simply pass `buoyancy = BuoyancyTracer()` to the
model constructor. Keep in mind that this treatment of buoyancy requires that `:b` be included as a
tracer, so it needs to be explicitly specified. For example, for a `NonhydrostaticModel`:

```jldoctest buoyancy
julia> model = NonhydrostaticModel(grid=grid, buoyancy=BuoyancyTracer(), tracers=:b)
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: (:b,)
├── closure: Nothing
├── buoyancy: BuoyancyTracer
└── coriolis: Nothing
```

We follow the same pattern to create a `HydrostaticFreeSurfaceModel` with buoyancy as a tracer:

```jldoctest buoyancy; filter = [r".*┌ Warning.*", r".*└ @ Oceananigans.*"]
julia> model = HydrostaticFreeSurfaceModel(grid=grid, buoyancy=BuoyancyTracer(), tracers=:b)
┌ Warning: HydrostaticFreeSurfaceModel is experimental. Use with caution!
└ @ Oceananigans.Models.HydrostaticFreeSurfaceModels ~/builds/tartarus-3/clima/oceananigans/src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl:106
HydrostaticFreeSurfaceModel{CPU, Float64}(time = 0 seconds, iteration = 0) 
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: (:b,)
├── closure: Nothing
├── buoyancy: Buoyancy{BuoyancyTracer, Oceananigans.Grids.ZDirection}
└── coriolis: Nothing
```



## Seawater buoyancy

To evolve temperature ``T`` and salinity ``S`` and diagnose the buoyancy, you can pass
`buoyancy = SeawaterBuoyancy()` to the constructor. This treatment of buoyancy requires that 
temperature ``T`` and salinity ``S`` be advected as tracers. For example, we can create a
`NonhydrostaticModel` with this option as (note that `NonhydrostaticModel` advects no tracers by
default, so we need to explicitly pass `tracers=(:T, :S)` to the constructor):

```jldoctest buoyancy
julia> model = NonhydrostaticModel(grid=grid, buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: (:T, :S)
├── closure: Nothing
├── buoyancy: SeawaterBuoyancy{Float64, LinearEquationOfState{Float64}, Nothing, Nothing}
└── coriolis: Nothing
```

We can similarly create a `HydrostaticFreeSurfaceModel` with the same treatment (in this case the
tracers don't need to be explicitly defined since this is default option for
`HydrostaticFreeSurfaceModel`):

```jldoctest buoyancy; filter = [r".*┌ Warning.*", r".*└ @ Oceananigans.*"]
julia> model = HydrostaticFreeSurfaceModel(grid=grid, buoyancy=SeawaterBuoyancy())
┌ Warning: HydrostaticFreeSurfaceModel is experimental. Use with caution!
└ @ Oceananigans.Models.HydrostaticFreeSurfaceModels ~/builds/tartarus-3/clima/oceananigans/src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl:106
HydrostaticFreeSurfaceModel{CPU, Float64}(time = 0 seconds, iteration = 0) 
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: (:T, :S)
├── closure: Nothing
├── buoyancy: Buoyancy{SeawaterBuoyancy{Float64, LinearEquationOfState{Float64}, Nothing, Nothing}, Oceananigans.Grids.ZDirection}
└── coriolis: Nothing
```

Without any options specified, a value of ``g = 9.80665 \, \text{m}\,\text{s}^{-2}`` is used for the gravitational
acceleration (corresponding to [standard gravity](https://en.wikipedia.org/wiki/Standard_gravity)) along
with a linear equation of state with thermal expansion and haline contraction coefficients suitable for seawater.

If, for example, you wanted to simulate fluids on another worlds, such as Europa, where ``g = 1.3 \, \text{m}\,\text{s}^{-2}``,
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

and similarly for a `HydrostaticFreeSurfaceModel`.


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


## Gravity alignment

If you want to simulate a set-up such that gravity doesn't align with the vertical (`z`) coordinate
(for example when simulating a tilted domain) you can wrap your option for buoyancy treatment in the
`Buoyancy()` function call, which allows for the option `vertical_unit_vector`. This is only
available for `NonhydrostaticModel`s, and it works for both `BuoyancyTracer()` and
`SeawaterBuoyancy()`. For example, a set-up in using buoyancy as a tracer and in which gravity is
tilted 45 degrees about the `x` axis can be done as

```jldoctest buoyancy
julia> θ = 45; # degrees

julia> g̃ = (0, sind(θ), cosd(θ));

julia> model = NonhydrostaticModel(grid=grid, 
                                   buoyancy=Buoyancy(model=BuoyancyTracer(), vertical_unit_vector=g̃), 
                                   tracers=:b)
NonhydrostaticModel{CPU, Float64}(time = 0 seconds, iteration = 0) 
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=64, Ny=64, Nz=64)
├── tracers: (:b,)
├── closure: Nothing
├── buoyancy: BuoyancyTracer
└── coriolis: Nothing
```


