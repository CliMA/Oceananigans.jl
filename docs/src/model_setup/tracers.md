# Tracers

The tracers to be advected around can be specified via a list of symbols. By default the model doesn't evolve any
tracers.

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest tracers
julia> grid = RectilinearGrid(size=(64, 64, 64), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid=grid)
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── timestepper: QuasiAdamsBashforth2TimeStepper
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing
```

But tracers can be added with the `tracers` keyword.
For example, to add conservative temperature `T` and absolute salinity `S`:

```jldoctest tracers
julia> grid = RectilinearGrid(size=(64, 64, 64), extent=(1, 1, 1));

julia> model = NonhydrostaticModel(grid=grid, tracers=(:T, :S))
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── timestepper: QuasiAdamsBashforth2TimeStepper
├── tracers: (T, S)
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing
```

whose fields can be accessed via `model.tracers.T` and `model.tracers.S`.

```jldoctest tracers
julia> model.tracers.T
64×64×64 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── boundary conditions: west=Periodic, east=Periodic, south=Periodic, north=Periodic, bottom=ZeroFlux, top=ZeroFlux, immersed=ZeroFlux
└── data: 66×66×66 OffsetArray(::Array{Float64, 3}, 0:65, 0:65, 0:65) with eltype Float64 with indices 0:65×0:65×0:65
    └── max=0.0, min=0.0, mean=0.0

julia> model.tracers.S
64×64×64 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── boundary conditions: west=Periodic, east=Periodic, south=Periodic, north=Periodic, bottom=ZeroFlux, top=ZeroFlux, immersed=ZeroFlux
└── data: 66×66×66 OffsetArray(::Array{Float64, 3}, 0:65, 0:65, 0:65) with eltype Float64 with indices 0:65×0:65×0:65
    └── max=0.0, min=0.0, mean=0.0
```

An arbitrary number of tracers may be simulated. For example, to simulate
``C_1``, ``CO₂``, and `nitrogen` as additional passive tracers,

```jldoctest tracers
julia> model = NonhydrostaticModel(grid=grid, tracers=(:T, :S, :C₁, :CO₂, :nitrogen))
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 64×64×64 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── timestepper: QuasiAdamsBashforth2TimeStepper
├── tracers: (T, S, C₁, CO₂, nitrogen)
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing
```

!!! info "Active versus passive tracers"
    An active tracer is a tracer whose distribution affects the evolution of momentum and other tracers.
    Typical ocean models evolve conservative temperature and absolute salinity as active tracers,
    which effect momentum through buoyancy forces.
    Passive tracers are "passive" in the sense that their distribution does not affect
    the evolution of other tracers or flow quantities.
