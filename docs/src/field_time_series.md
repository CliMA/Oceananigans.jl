# FieldTimeSeries

A `FieldTimeSeries` is a data structure that implements a `Field` with time dependence.

## Basic usage

A `FieldTimeSeries` may be constructed directly on an existing grid, similarly to a `Field`:

```jldoctest field_time_series
grid = RectilinearGrid(; topology = (Periodic, Periodic, Bounded),
                         size = (8, 8, 8),
                         extent = (2π, 2π, 2π))
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

u₀(x, y, z) = sin(x) * sin(y)
v₀(x, y, z) = cos(x) * cos(y)
set!(model, u=u₀, v=v₀)

simulation = Simulation(model; Δt=0.1, stop_iteration=10)

simulation.output_writers[:fields] = JLD2Writer(model, model.velocities;
                                                schedule = IterationInterval(1),
                                                filename = "test.jld2",
                                                overwrite_existing = true)
run!(simulation)

fts = FieldTimeSeries("test.jld2", "u")

# output
8×8×8×11 FieldTimeSeries{InMemory} located at (Face, Center, Center) of u at test.jld2
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── indices: (:, :, :)
├── time_indexing: Linear()
├── backend: InMemory()
├── path: test.jld2
├── name: u
└── data: 14×14×14×11 OffsetArray(::Array{Float64, 4}, -2:11, -2:11, -2:11, 1:11) with eltype Float64 with indices -2:11×-2:11×-2:11×1:11
    └── max=0.92388, min=-0.92388, mean=-6.98855e-20
```

In either case, it can be indexed as if it were a four-dimensional `Field`, with time as the last index:

```jldoctest field_time_series
fts[4, 4, 4, 4]

# output
0.26985207200050354
```

```jldoctest field_time_series
interior(fts, :, 1, 1, :) # x-t array

# output
8×11 view(::Array{Float64, 4}, 4:11, 4, 4, :) with eltype Float64:
 -2.45381e-18   1.98439e-17   1.07381e-17  -3.79997e-18  …   5.0393e-17    8.05027e-17   9.77895e-17   9.03376e-17
  0.270598      0.270338      0.27009       0.269852         0.268994      0.2688        0.268613      0.268432
  0.382683      0.382608      0.382528      0.382444         0.382067      0.381964      0.381859      0.381751
  0.270598      0.270338      0.27009       0.269852         0.268994      0.2688        0.268613      0.268432
  1.84987e-18   3.56235e-17   1.05233e-17   5.62883e-17      4.16375e-17  -1.16443e-17  -3.00039e-17   3.23178e-17
 -0.270598     -0.270338     -0.27009      -0.269852     …  -0.268994     -0.2688       -0.268613     -0.268432
 -0.382683     -0.382608     -0.382528     -0.382444        -0.382067     -0.381964     -0.381859     -0.381751
 -0.270598     -0.270338     -0.27009      -0.269852        -0.268994     -0.2688       -0.268613     -0.268432
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
    └── max=0.92388, min=-0.92388, mean=0.0
```

In place of an index, an `Oceananigans.Units.Time` can be used to interpolate values at a given time.

```jldoctest field_time_series
using Oceananigans.Units: Time

fts[4, 4, 4, Time(0.33)] # 0.2697837561368942
fts[Time(0.33)]

# output
8×8×8 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: BinaryOperation at (Face, Center, Center)
├── status: Oceananigans.Fields.FixedTime{Float64}
└── data: 14×14×14 OffsetArray(::Array{Float64, 3}, -2:11, -2:11, -2:11) with eltype Float64 with indices -2:11×-2:11×-2:11
    └── max=0.922644, min=-0.922644, mean=-1.0842e-18
```

Contained data may be manipulated with [`set!`](@ref). Providing a single time index will simply pass [`set!`](@ref) to the corresponding `Field`:

```jldoctest field_time_series
set!(fts, (x, y, z)->2x, 1) # equivalent to set!(fts[1], (x, y, z)->2x)

# output
8×8×8 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 14×14×14 OffsetArray(view(::Array{Float64, 4}, :, :, :, 1), -2:11, -2:11, -2:11) with eltype Float64 with indices -2:11×-2:11×-2:11
    └── max=10.9956, min=0.0, mean=5.49779
```

Providing a vector of `Field`s will copy the data in each `Field` to the `FieldTimeSeries`:

```jldoctest field_time_series
fields = [Field{Face, Center, Center}(grid) for n in 1:length(fts)]
set!(fts, fields)

# output
8×8×8×11 FieldTimeSeries{InMemory} located at (Face, Center, Center) of u at test.jld2
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── indices: (:, :, :)
├── time_indexing: Linear()
├── backend: InMemory()
├── path: test.jld2
├── name: u
└── data: 14×14×14×11 OffsetArray(::Array{Float64, 4}, -2:11, -2:11, -2:11, 1:11) with eltype Float64 with indices -2:11×-2:11×-2:11×1:11
    └── max=0.0, min=0.0, mean=0.0
```

## Extrapolated data

`FieldTimeSeries` supports multiple methods of time indexing outside of the given times.

```jldoctest field_time_series
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

Each of the above `FieldTimeSeries` contain identical underlying data, however their behavior differs when indexed using `Oceananigans.Units.Time` outside of the range of given times.

```jldoctest field_time_series
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

## OnDisk FieldTimeSeries

The default `backend` for a `FieldTimeSeries` is `InMemory`. This is sufficient for small timeseries, but large simulation data likely cannot all be loaded into memory. Instead, it is possible to lazily load a timeseries from storage by passing `backend = OnDisk()` to `FieldTimeSeries`:

```jldoctest field_time_series
ondisk_fts = FieldTimeSeries("test.jld2", "u"; backend=OnDisk())

# output
8×8×8×11 FieldTimeSeries{OnDisk} located at (Face, Center, Center) of u at test.jld2
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── indices: (:, :, :)
├── time_indexing: Linear()
├── backend: OnDisk
├── path: test.jld2
└── name: u
```

Data is only loaded when indexed with an integer or time:
```jldoctest field_time_series
ondisk_fts[1]

# output
8×8×8 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 14×14×14 OffsetArray(::Array{Float32, 3}, -2:11, -2:11, -2:11) with eltype Float32 with indices -2:11×-2:11×-2:11
    └── max=0.92388, min=-0.92388, mean=-1.74623e-10
```

```jldoctest field_time_series
ondisk_fts[Time(0.3)]

# output
8×8×8 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
├── operand: BinaryOperation at (Face, Center, Center)
├── status: Oceananigans.Fields.FixedTime{Float64}
└── data: 14×14×14 OffsetArray(::Array{Float64, 3}, -2:11, -2:11, -2:11) with eltype Float64 with indices -2:11×-2:11×-2:11
    └── max=0.922754, min=-0.922754, mean=2.1684e-19
```

!!! note "Lazily loading multiple time steps"
    Using `backend = InMemory(N)` will create a `FieldTimeSeries` that lazily loads windows of length `N` time steps. When indexed outside the currently loaded window, a window containing the requested time step and the following `N-1` time steps is loaded into memory.

An empty `OnDisk` `FieldTimeSeries` can be created by also including `path` and `name` keywords. These can only be [`set!`](@ref) with a single `Field` and integer index. Doing so will write the data to storage if that index doesn't exist, creating a file if necessary.

```jldoctest field_time_series
new_ondisk_fts = FieldTimeSeries{Center, Center, Center}(grid, times;
                                                         path = "new.jld2",
                                                         name = "c",
                                                         backend=OnDisk())

field = Field{Center, Center, Center}(grid)
for (n, t) in enumerate(times)
    set!(field, (x, y, z) -> x * y * z * t)

    # Write the field to disk
    set!(new_ondisk_fts, field, n)
end

# Read the data
new_fts = FieldTimeSeries("new.jld2", "c")

# output
8×8×8×11 FieldTimeSeries{InMemory} located at (Center, Center, Center) of c at new.jld2
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── indices: (:, :, :)
├── time_indexing: Linear()
├── backend: InMemory()
├── path: new.jld2
├── name: c
└── data: 14×14×14×11 OffsetArray(::Array{Float64, 4}, -2:11, -2:11, -2:11, 1:11) with eltype Float64 with indices -2:11×-2:11×-2:11×1:11
    └── max=0.0, min=-8.23975, mean=-0.118738
```

!!! warn "Modifying simulation output files"
    This behaviour is intended for outputting fields to new files. It is not recommended to use a `FieldTimeSeries` to modify simulation output. Indexing may not be compatible.

# FieldDataset

A `FieldDataset` is essentially a container for a collection of `FieldTimeSeries`, together with some metadata. These may be used to read an entire simulation's worth of output:

```jldoctest field_time_series
fds = FieldDataset("test.jld2"; backend=InMemory())

# output
FieldDataset with 3 fields and 0 metadata entries:
├── v: 8×8×8×11 FieldTimeSeries{InMemory} located at (Center, Face, Center) of v at test.jld2
├── w: 8×8×9×11 FieldTimeSeries{InMemory} located at (Center, Center, Face) of w at test.jld2
└── u: 8×8×8×11 FieldTimeSeries{InMemory} located at (Face, Center, Center) of u at test.jld2
```

An empty `FieldDataset` may also be constructed by providing a grid, saved times and a tuple of field names. Optional keyword arguments specify locations, indices, boundary conditions and output path (in the case of `backend = OnDisk()`):

```jldoctest field_time_series
grid = RectilinearGrid(; topology = (Periodic, Periodic, Bounded),
                         size = (8, 8, 8),
                         extent = (1, 1, 1))
times = 0:10
fields = (:u, :v)
location = (; u = (Face(), Center(), Center()),
              v = (Center(), Face(), Center()))

new_fds = FieldDataset(grid, times, fields; location, backend=InMemory())

# output
FieldDataset with 2 fields and 0 metadata entries:
├── v: 8×8×8×11 FieldTimeSeries{InMemory} located at (Center, Face, Center) on CPU
└── u: 8×8×8×11 FieldTimeSeries{InMemory} located at (Face, Center, Center) on CPU
```

A convenience constructor also exists to generate a `FieldDataset` according to a `NamedTuple` of preexisting fields. The following is equivalent to the above:

```jldoctest field_time_series
grid = RectilinearGrid(; topology = (Periodic, Periodic, Bounded),
                         size = (8, 8, 8),
                         extent = (1, 1, 1))
times = 0:10

u = Field{Face, Center, Center}(grid)
v = Field{Center, Face, Center}(grid)
fields = (; u, v)

new_fds = FieldDataset(times, fields; backend=InMemory())

# output
FieldDataset with 2 fields and 0 metadata entries:
├── v: 8×8×8×11 FieldTimeSeries{InMemory} located at (Center, Face, Center) on CPU
└── u: 8×8×8×11 FieldTimeSeries{InMemory} located at (Face, Center, Center) on CPU
```

Note that the new `FieldDataset` is unrelated to the data contained in the input fields. The result inherits the grid, locations, indices and boundary conditions of the input fields. Individual timeseries may be retrieved by indexing

```jldoctest field_time_series
fds.u

# output
8×8×8×11 FieldTimeSeries{InMemory} located at (Face, Center, Center) of u at test.jld2
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── indices: (:, :, :)
├── time_indexing: Linear()
├── backend: InMemory()
├── path: test.jld2
├── name: u
└── data: 14×14×14×11 OffsetArray(::Array{Float64, 4}, -2:11, -2:11, -2:11, 1:11) with eltype Float64 with indices -2:11×-2:11×-2:11×1:11
    └── max=1.12648, min=-0.0388058, mean=0.357269
```

Calling [`set!`](@ref) on a `FieldDataset` with keyword arguments will pass [`set!`](@ref) to each of the contained `FieldTimeSeries`, which will behave according to their backend. The following sets data in u and v in the first time index to u_func and v_func respectively:

```jldoctest field_time_series
u_func(x, y, z) = -x
v_func(x, y, z) = y

set!(new_fds, 1; u=u_func, v=v_func)

println(new_fds.u[1], "\n\n", new_fds.v[1])

# output
8×8×8 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 14×14×14 OffsetArray(view(::Array{Float64, 4}, :, :, :, 1), -2:11, -2:11, -2:11) with eltype Float64 with indices -2:11×-2:11×-2:11
    └── max=-0.0, min=-0.875, mean=-0.4375

8×8×8 Field{Center, Face, Center} on RectilinearGrid on CPU
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 14×14×14 OffsetArray(view(::Array{Float64, 4}, :, :, :, 1), -2:11, -2:11, -2:11) with eltype Float64 with indices -2:11×-2:11×-2:11
    └── max=0.875, min=0.0, mean=0.4375
```

# Post-processing

When running simulations, it may be preferrable to calculate diagnostics after the simulation has completed. The input and output capabilities of `FieldTimeSeries`, together with `AbstractOperations` on fields permits a straight-forward post-processing pipeline for the offline computation of fields. The following example reads the data contained in `test.jld2` and calculates the kinetic energy for each timestep, which is then output to `test_ke.jld2`

```jldoctest field_time_series
input_fds = FieldDataset("test.jld2"; backend=OnDisk())
times = input_fds.u.times

# Initialise containers for input fields
u = Field(input_fds.u[1])
v = Field(input_fds.v[1])
w = Field(input_fds.w[1])

# Create output fields
ke_density = Field((u^2 + v^2 + w^2) / 2)
ke = Field(Integral(ke_density))
output_fields = (; ke_density, ke)

# Create output
output_fds = FieldDataset(times, output_fields; backend=OnDisk(), path="test_ke.jld2")

# Loop over outputted iterations
for n in 1:length(times)
    # Read current timestep
    set!(u, input_fds.u[n])
    set!(v, input_fds.v[n])
    set!(w, input_fds.w[n])

    # Calculate fields
    compute!(ke)

    # Output
    set!(output_fds, n; ke_density, ke)
end

ke_timeseries = FieldTimeSeries("test_ke.jld2", "ke")

ke_timeseries[1, 1, 1, :]

# output
11-element Vector{Float64}:
 62.012554977702926
 61.97056180705332
 61.92860328861815
 61.88667457108706
 61.84476710691496
 61.80288551639598
 61.761035574899566
 61.71920688676128
 61.677401993144294
 61.63562043201756
 61.59385180771814
```
