# FieldTimeSeries

A `FieldTimeSeries` is a data structure that implements a `Field` with time dependence.

## Basic usage

A `FieldTimeSeries` may be constructed directly on an existing grid, similarly to a `Field`:

```jldoctest field_time_series
grid = RectilinearGrid(; topology = (Periodic, Periodic, Bounded),
                         size = (8, 8, 8),
                         extent = (1, 1, 1))
times = 0:10
fts = FieldTimeSeries{Center, Center, Center}(grid, times)

# output
8×8×8×11 FieldTimeSeries{InMemory} located at (Center, Center, Center) on CPU
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── indices: (:, :, :)
├── time_indexing: Clamp()
├── backend: InMemory()
└── data: 14×14×14×11 OffsetArray(::Array{Float64, 4}, -2:11, -2:11, -2:11, 1:11) with eltype Float64 with indices -2:11×-2:11×-2:11×1:11
    └── max=0.0, min=0.0, mean=0.0
```

Additionally, a `FieldTimeSeries` may be generated from simulation output:

```jldoctest field_time_series
model = NonhydrostaticModel(grid; advection=WENO())
ϵ(x, y, z) = rand()
set!(model, u=ϵ, v=ϵ, w=ϵ)
simulation = Simulation(model; Δt=0.01, stop_iteration=100)

simulation.output_writers[:fields] = JLD2Writer(model, model.velocities,
                                                schedule = TimeInterval(0.1),
                                                filename = "test.jld2",
                                                overwrite_existing = true)
run!(simulation)

fts = FieldTimeSeries("test.jld2", "u")

# output
8×8×8×10 FieldTimeSeries{InMemory} located at (Face, Center, Center) of u at test.jld2
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── indices: (:, :, :)
├── time_indexing: Linear()
├── backend: InMemory()
├── path: test.jld2
├── name: u
└── data: 14×14×14×10 OffsetArray(::Array{Float64, 4}, -2:11, -2:11, -2:11, 1:10) with eltype Float64 with indices -2:11×-2:11×-2:11×1:10
    └── max=1.07545, min=-0.0808614, mean=0.345513
```

In either case, it can be indexed as if it were a four-dimensional `Field`, with time as the last index:

```jldoctest field_time_series
fts[4, 4, 4, 4]

# output
0.5358667969703674
```

```jldoctest field_time_series
interior(fts, :, 1, 1, :) # x-t array

# output
8×10 view(::Array{Float64, 4}, 4:11, 4, 4, :) with eltype Float64:
 0.860741  0.763312  0.61371   0.510262  0.498051  0.498372  0.494729  0.476513  0.454274  0.441559
 0.18571   0.460542  0.527368  0.496212  0.450224  0.448384  0.457524  0.46419   0.460685  0.449141
 0.234925  0.356419  0.455896  0.501329  0.501862  0.485451  0.47594   0.467415  0.463953  0.459797
 0.562129  0.488347  0.461392  0.466226  0.482304  0.495795  0.502234  0.49983   0.492295  0.483445
 0.314     0.447676  0.47347   0.47389   0.476044  0.482464  0.492456  0.50166   0.50391   0.501611
 0.38311   0.403706  0.444067  0.465936  0.467379  0.466081  0.465873  0.474316  0.484465  0.492067
 0.317062  0.339962  0.402139  0.438216  0.452982  0.452426  0.446957  0.443386  0.45102   0.465696
 0.693166  0.508454  0.469629  0.491742  0.493768  0.478989  0.459651  0.441641  0.432787  0.435327
```

Providing just one index returns a `Field` containing a view of the data at the corresponding time step:

```jldoctest field_time_series
fts[1]

# output
8×8×8 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 14×14×14 OffsetArray(view(::Array{Float64, 4}, :, :, :, 1), -2:11, -2:11, -2:11) with eltype Float64 with indices -2:11×-2:11×-2:11
    └── max=1.07545, min=-0.0808614, mean=0.489556
```

In place of an index, an `Oceananigans.Units.Time` can be used to interpolate values at a given time.

```jldoctest field_time_series
using Oceananingans.Units: Time

fts[4, 4, 4, Time(0.33)]
fts[Time(0.33)]
```

Contained data may be manipulated with `set!`. Providing a single time index will simply pass `set!` to the corresponding `Field`:

```jldoctest field_time_series
set!(fts, (x, y, z)->2x, 1) # equivalent to set!(fts[1], (x, y, z)->2x)

# output
8×8×8 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 14×14×14 OffsetArray(view(::Array{Float64, 4}, :, :, :, 1), -2:11, -2:11, -2:11) with eltype Float64 with indices -2:11×-2:11×-2:11
    └── max=1.75, min=0.0, mean=0.875
```

Providing a vector of `Field`s will copy the data in each `Field` to the `FieldTimeSeries`:

```jldoctest field_time_series
fields = [Field{Face, Center, Center}(grid) for n in 1:length(fts)]
set!(fts, fields)
fts

# output
8×8×8×10 FieldTimeSeries{InMemory} located at (Face, Center, Center) of u at test.jld2
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── indices: (:, :, :)
├── time_indexing: Linear()
├── backend: InMemory()
├── path: test.jld2
├── name: u
└── data: 14×14×14×10 OffsetArray(::Array{Float64, 4}, -2:11, -2:11, -2:11, 1:10) with eltype Float64 with indices -2:11×-2:11×-2:11×1:10
    └── max=0.0, min=0.0, mean=0.0
```

## Extrapolated data
`FieldTimeSeries` supports multiple methods of time indexing outside of the given times.

```jldoctext field_time_series
using Oceananigans.OutputReaders: Clamp, Linear, Cyclical

grid = RectilinearGrid(; size=(), extent=(), topology=(Flat, Flat, Flat))

fts_clamp = FieldTimeSeries{Center, Center, Center}(grid, [0, 1]; time_indexing=Clamp())
fts_linear = FieldTimeSeries{Center, Center, Center}(grid, [0, 1]; time_indexing=Linear())
fts_cyclical = FieldTimeSeries{Center, Center, Center}(grid, [0, 1]; time_indexing=Cyclical())

for fts in (fts_clamp, fts_linear, fts_cyclical)
    set!(fts, 0, 1)
    set!(fts, 1, 2)
end
```

Each of the above `FieldTimeSeries` contain identical underlying data, however their behaviour differs when indexed using `Oceananigans.Units.Time` outside of the range of given times.

```jldoctext field_time_series
println("t | Clamp | Linear | Cyclical")
for t in 0:0.5:3
    clamp = fts_clamp[1, 1, 1, Time(t)]
    linear = fts_linear[1, 1, 1, Time(t)]
    cyclical = fts_cyclical[1, 1, 1, Time(t)]

    println("$t   | $clamp   | $linear    | $cyclical")
end

# output
t   | Clamp | Linear | Cyclical
0.0 | 0.0   | 0.0    | 0.0
0.5 | 0.5   | 0.5    | 0.5
1.0 | 1.0   | 1.0    | 1.0
1.5 | 1.0   | 1.5    | 0.5
2.0 | 1.0   | 2.0    | 0.0
2.5 | 1.0   | 2.5    | 0.5
3.0 | 1.0   | 3.0    | 1.0
```

