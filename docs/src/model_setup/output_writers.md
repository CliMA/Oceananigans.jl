# Output writers

`AbstractOutputWriter`s save data to disk.
`Oceananigans` provides three ways to write output:

1. [`NetCDFWriter`](@ref) for output of arrays and scalars that uses [NCDatasets.jl](https://github.com/Alexander-Barth/NCDatasets.jl)
2. [`JLD2Writer`](@ref) for arbitrary julia data structures that uses [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)
3. [`Checkpointer`](@ref) that automatically saves as much model data as possible, using [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)

The `Checkpointer` is discussed in detail on a separate [section](@ref checkpointing) of the documentation.

## Basic usage

[`NetCDFWriter`](@ref) and [`JLD2Writer`](@ref) require four inputs:

1. The `model` from which output data is sourced (required to initialize the `OutputWriter`).
2. A key-value pairing of output "names" and "output" objects. `JLD2Writer` accepts `NamedTuple`s and `Dict`s;
   `NetCDFWriter` accepts `Dict`s with string-valued keys. Output objects are either `AbstractField`s or
   functions that return data when called via `func(model)`.
3. A `schedule` on which output is written. `TimeInterval`, `IterationInterval`, `WallTimeInterval` schedule
   periodic output according to the simulation time, simulation interval, or "wall time" (the physical time
   according to a clock on your wall). A fourth `schedule` called `AveragedTimeInterval` specifies
   periodic output that is time-averaged over a `window` prior to being written.
4. The `filename` and `dir`ectory.

Other important keyword arguments are

* `indices` for outputting subregions, two- and one-dimensional slices of fields. Specifies the indices to write to disk with a `Tuple` of `Colon`, `UnitRange`,or `Int` elements. For example, `indices = (:, :, 1)` implies outputing ``x-y``-slices of the bottom-most index (`k=1`). Defaults to `(:, :, :)`, i.e., "all indices".
* `with_halos :: Boolean`: whether to output the halos (`true`) or only the interior points (`false`; default).

* `array_type` for specifying the type of the array that holds outputted field data. The default is
  `Array{Float64}`, or arrays of single-precision floating point numbers.

Once an `OutputWriter` is created, it can be used to write output by adding it the
ordered dictionary `simulation.output_writers`. prior to calling `run!(simulation)`.

More specific detail about the `NetCDFWriter` and `JLD2Writer` is given below.

!!! tip "Time step alignment and output writing"
    Oceananigans simulations will shorten the time step as needed to align model output with each
    output writer's schedule.

## NetCDF output writer

Model data can be saved to NetCDF files along with associated metadata. The NetCDF output writer is generally used by
passing it a dictionary of (label, field) pairs and any indices for slicing if you don't want to save the full 3D field.

### Examples

Saving the u velocity field and temperature fields as full 3D fields, surface 2D slices, and
1D columns to separate NetCDF files:

```@example netcdf1
using Oceananigans
using NCDatasets

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))

model = NonhydrostaticModel(grid=grid, tracers=:c)

simulation = Simulation(model, Δt=12, stop_time=3600)

fields = Dict("u" => model.velocities.u, "c" => model.tracers.c)

simulation.output_writers[:field_writer] =
    NetCDFWriter(model, fields, filename="more_fields.nc", schedule=TimeInterval(60))
```

```@example netcdf1
simulation.output_writers[:surface_slice_writer] =
    NetCDFWriter(model, fields, filename="another_surface_xy_slice.nc",
                 schedule=TimeInterval(60), indices=(:, :, grid.Nz))
```

```@example netcdf1
simulation.output_writers[:averaged_profile_writer] =
    NetCDFWriter(model, fields,
                 filename = "another_averaged_z_profile.nc",
                 schedule = AveragedTimeInterval(60, window=20),
                 indices = (1, 1, :))
```

`NetCDFWriter` also accepts output functions that write scalars and arrays to disk,
provided that their `dimensions` are provided:

```@example
using Oceananigans
using NCDatasets

Nx, Ny, Nz = 16, 16, 16

grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(1, 2, 3))

model = NonhydrostaticModel(grid=grid)

simulation = Simulation(model, Δt=1.25, stop_iteration=3)

f(model) = model.clock.time^2; # scalar output

g(model) = model.clock.time .* exp.(znodes(grid, Center())) # single-column profile output (vector)

xC, yF = xnodes(grid, Center()), ynodes(grid, Face())

XC = [xC[i] for i in 1:Nx, j in 1:Ny]
YF = [yF[j] for i in 1:Nx, j in 1:Ny]

h(model) = @. model.clock.time * sin(XC) * cos(YF) # x-y slice output (2D array)

outputs = Dict("scalar" => f, "profile" => g, "slice" => h)

dims = Dict("scalar" => (), "profile" => ("z_aac",), "slice" => ("x_caa", "y_aca"))

output_attributes = Dict(
    "scalar"  => Dict("longname" => "Some scalar", "units" => "bananas"),
    "profile" => Dict("longname" => "Some vertical profile", "units" => "watermelons"),
    "slice"   => Dict("longname" => "Some slice", "units" => "mushrooms"))

global_attributes = Dict("location" => "Bay of Fundy", "onions" => 7)

simulation.output_writers[:things] =
    NetCDFWriter(model, outputs,
                 schedule=IterationInterval(1), filename="things.nc", dimensions=dims, verbose=true,
                 global_attributes=global_attributes, output_attributes=output_attributes)
```

`NetCDFWriter` can also be configured for `outputs` that are interpolated or regridded
to a different grid than `model.grid`. To use this functionality, include the keyword argument
`grid = output_grid`.

```@example
using Oceananigans
using Oceananigans.Fields: interpolate!
using NCDatasets

grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 1));
model = NonhydrostaticModel(; grid)

coarse_grid = RectilinearGrid(size=(grid.Nx, grid.Ny, grid.Nz÷2), extent=(grid.Lx, grid.Ly, grid.Lz))
coarse_u = Field{Face, Center, Center}(coarse_grid)

interpolate_u(model) = interpolate!(coarse_u, model.velocities.u)
outputs = (; u = interpolate_u)

output_writer = NetCDFWriter(model, outputs;
                             grid = coarse_grid,
                             filename = "coarse_u.nc",
                             schedule = IterationInterval(1))
```

See [`NetCDFWriter`](@ref) for more information.

## JLD2 output writer

JLD2 is a fast HDF5 compatible file format written in pure Julia.
JLD2 files can be opened in Julia with the [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) package
and in Python with the [h5py](https://www.h5py.org/) package.

The `JLD2Writer` receives either a `Dict`ionary or `NamedTuple` containing
`name, output` pairs. The `name` can be a symbol or string. The `output` must either be
an `AbstractField` or a function called with `func(model)` that returns arbitrary output.
Whenever output needs to be written, the functions will be called and the output
of the function will be saved to the JLD2 file.

### Examples

Write out 3D fields for u, v, w, and a tracer c, along with a horizontal average:

```@example jld2_output_writer
using Oceananigans
using Oceananigans.Utils: hour, minute

model = NonhydrostaticModel(grid=RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)), tracers=(:c,))
simulation = Simulation(model, Δt=12, stop_time=1hour)

function init_save_some_metadata!(file, model)
    file["author"] = "Chim Riggles"
    file["parameters/coriolis_parameter"] = 1e-4
    file["parameters/density"] = 1027
    return nothing
end

c_avg = Field(Average(model.tracers.c, dims=(1, 2)))

# Note that model.velocities is NamedTuple
simulation.output_writers[:velocities] = JLD2Writer(model, model.velocities,
                                                    filename = "some_more_data.jld2",
                                                    schedule = TimeInterval(20minute),
                                                    init = init_save_some_metadata!)
```

and a time- and horizontal-average of tracer `c` every 20 minutes of simulation time
to a file called `some_more_averaged_data.jld2`

```@example jld2_output_writer
simulation.output_writers[:avg_c] = JLD2Writer(model, (; c=c_avg),
                                               filename = "some_more_averaged_data.jld2",
                                               schedule = AveragedTimeInterval(20minute, window=5minute))
```

See [`JLD2Writer`](@ref) for more information.

## Time-averaged output

Time-averaged output is specified by setting the `schedule` keyword argument for either `NetCDFWriter` or
`JLD2Writer` to [`AveragedTimeInterval`](@ref).

With `AveragedTimeInterval`, the time-average of ``a`` is taken as a left Riemann sum corresponding to

```math
\langle a \rangle = \frac{1}{T} \int_{t_i-T}^{t_i} a \, \mathrm{d} t \, ,
```

where ``\langle a \rangle`` is the time-average of ``a``, ``T`` is the time-`window` for averaging specified by
the `window` keyword argument to `AveragedTimeInterval`, and the ``t_i`` are discrete times separated by the
time `interval`. The ``t_i`` specify both the end of the averaging window and the time at which output is written.

### Example

Building an `AveragedTimeInterval` that averages over a 1 day window, every 4 days,

```jldoctest averaged_time_interval
using Oceananigans
using Oceananigans.Units

schedule = AveragedTimeInterval(4days, window=1day)

# output
AveragedTimeInterval(window=1 day, stride=1, interval=4 days)
```

An `AveragedTimeInterval` schedule directs an output writer
to time-average its outputs before writing them to disk:

```@example averaged_time_interval
using Oceananigans
using Oceananigans.Units

model = NonhydrostaticModel(grid=RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=10minutes, stop_time=30days)

simulation.output_writers[:velocities] = JLD2Writer(model, model.velocities,
                                                    filename = "even_more_averaged_velocity_data.jld2",
                                                    schedule = AveragedTimeInterval(4days, window=1day, stride=2))
```
