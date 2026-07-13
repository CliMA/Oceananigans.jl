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

simulation.output_writers[:fields] = JLD2Writer(model, model.velocities;
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

Contained data may be manipulated with [`set!`](@ref). Providing a single time index will simply pass [`set!`](@ref) to the corresponding `Field`:

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

Each of the above `FieldTimeSeries` contain identical underlying data, however their behaviour differs when indexed using `Oceananigans.Units.Time` outside of the range of given times.

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
8×8×8×10 FieldTimeSeries{OnDisk} located at (Face, Center, Center) of u at test.jld2
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
    └── max=1.12648, min=-0.0388058, mean=0.502443
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
    └── max=0.758505, min=0.18855, mean=0.502443
```

An empty `OnDisk` `FieldTimeSeries` can be created by also including `path` and `name` keywords. These can only be [`set!`](@ref) with a single `Field` and integer index. Doing so will write the data to storage if that index doesn't exist, creating a file if necessary.

```jldoctest field_time_series
new_ondisk_fts = FieldTimeSeries{Center, Center, Center}(grid, times; 
                                                         path = "new.jld2",
                                                         name = "c",
                                                         backend=OnDisk()
                                                        )

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
                         extent = (1, 1, 1)
                      )
times = 0:10
fields = (:u, :v)
location = (; u = (Face(), Center(), Center()), 
              v = (Center(), Face(), Center())
            )

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
                         extent = (1, 1, 1)
                      )
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
for i in 1:length(times)
    # Read current timestep
    set!(u, input_fds.u[i])
    set!(v, input_fds.v[i])
    set!(w, input_fds.w[i])

    # Calculate fields
    compute!(ke)

    # Output
    set!(output_fds, i; ke_density, ke)
end

ke_timeseries = FieldTimeSeries("test_ke.jld2", "ke")

ke_timeseries[1, 1, 1, :]

# output
10-element Vector{Float64}:
 0.6612136948970146
 0.5811033841455355
 0.5583625429135282
 0.549366099992767
 0.5451507281395607
 0.5428409611340612
 0.5413548710057512
 0.5403541878913529
 0.539588899933733
 0.5389546225778759
```