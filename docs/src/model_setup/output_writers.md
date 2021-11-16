# Output writers

`AbstractOutputWriter`s save data to disk.
`Oceananigans` provides three ways to write output:

1. [`NetCDFOutputWriter`](@ref) for output of arrays and scalars that uses [NCDatasets.jl](https://github.com/Alexander-Barth/NCDatasets.jl)
2. [`JLD2OutputWriter`](@ref) for arbitrary julia data structures that uses [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)
3. [`Checkpointer`](@ref) that automatically saves as much model data as possible, using [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)

The `Checkpointer` is discussed on a separate documentation page.

## Basic usage

[`NetCDFOutputWriter`](@ref) and [`JLD2OutputWriter`](@ref) require four inputs:

1. The `model` from which output data is sourced (required to initialize the `OutputWriter`).
2. A key-value pairing of output "names" and "output" objects. `JLD2OutputWriter` accepts `NamedTuple`s and `Dict`s;
   `NetCDFOutputWriter` accepts `Dict`s with string-valued keys. Output objects are either `AbstractField`s or
   functions that return data when called via `func(model)`.
3. A `schedule` on which output is written. `TimeInterval`, `IterationInterval`, `WallTimeInterval` schedule
   periodic output according to the simulation time, simulation interval, or "wall time" (the physical time
   according to a clock on your wall). A fourth `schedule` called `AveragedTimeInterval` specifies
   periodic output that is time-averaged over a `window` prior to being written.
4. The filename and directory. Currently `NetCDFOutputWriter` accepts one `filepath` argument, while
   `JLD2OutputWriter` accepts a filename `prefix` and `dir`ectory.

Other important keyword arguments are

* `field_slicer::FieldSlicer` for outputting subregions, two- and one-dimensional slices of fields.
  By default a `FieldSlicer` is used to remove halo regions from fields so that only the physical
  portion of model data is saved to disk.

* `array_type` for specifying the type of the array that holds outputted field data. The default is
  `Array{Float32}`, or arrays of single-precision floating point numbers.

Once an `OutputWriter` is created, it can be used to write output by adding it the
ordered dictionary `simulation.output_writers`. prior to calling `run!(simulation)`.

More specific detail about the `NetCDFOutputWriter` and `JLD2OutputWriter` is given below.

!!! tip "Time step alignment and output writing"
    Oceananigans simulations will shorten the time step as needed to align model output with each
    output writer's schedule.

## NetCDF output writer

Model data can be saved to NetCDF files along with associated metadata. The NetCDF output writer is generally used by
passing it a dictionary of (label, field) pairs and any indices for slicing if you don't want to save the full 3D field.

### Examples

Saving the u velocity field and temperature fields, the full 3D fields and surface 2D slices
to separate NetCDF files:

```jldoctest netcdf1
using Oceananigans

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1));

model = NonhydrostaticModel(grid=grid,);

simulation = Simulation(model, Δt=12, stop_time=3600);

fields = Dict("u" => model.velocities.u, "v" => model.velocities.v);

simulation.output_writers[:field_writer] =
    NetCDFOutputWriter(model, fields, filepath="more_fields.nc", schedule=TimeInterval(60))

# output
NetCDFOutputWriter scheduled on TimeInterval(1 minute):
├── filepath: more_fields.nc
├── dimensions: zC(16), zF(17), xC(16), yF(16), xF(16), yC(16), time(0)
├── 2 outputs: ["v", "u"]
├── field slicer: FieldSlicer(:, :, :, with_halos=false)
└── array type: Array{Float32}
```

```jldoctest netcdf1
simulation.output_writers[:surface_slice_writer] =
    NetCDFOutputWriter(model, fields, filepath="another_surface_xy_slice.nc",
                       schedule=TimeInterval(60), field_slicer=FieldSlicer(k=grid.Nz))

# output
NetCDFOutputWriter scheduled on TimeInterval(1 minute):
├── filepath: another_surface_xy_slice.nc
├── dimensions: zC(1), zF(1), xC(16), yF(16), xF(16), yC(16), time(0)
├── 2 outputs: ["v", "u"]
├── field slicer: FieldSlicer(:, :, 16, with_halos=false)
└── array type: Array{Float32}
```

```jldoctest netcdf1
simulation.output_writers[:averaged_profile_writer] =
    NetCDFOutputWriter(model, fields,
                       filepath = "another_averaged_z_profile.nc",
                       schedule = AveragedTimeInterval(60, window=20),
                       field_slicer = FieldSlicer(i=1, j=1))

# output
NetCDFOutputWriter scheduled on TimeInterval(1 minute):
├── filepath: another_averaged_z_profile.nc
├── dimensions: zC(16), zF(17), xC(1), yF(1), xF(1), yC(1), time(0)
├── 2 outputs: ["v", "u"] averaged on AveragedTimeInterval(window=20 seconds, stride=1, interval=1 minute)
├── field slicer: FieldSlicer(1, 1, :, with_halos=false)
└── array type: Array{Float32}
```

`NetCDFOutputWriter` also accepts output functions that write scalars and arrays to disk,
provided that their `dimensions` are provided:

```jldoctest
using Oceananigans

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 2, 3));

model = NonhydrostaticModel(grid=grid);

simulation = Simulation(model, Δt=1.25, stop_iteration=3);

f(model) = model.clock.time^2; # scalar output
g(model) = model.clock.time .* exp.(znodes(Center, grid)); # vector/profile output
h(model) = model.clock.time .* (   sin.(xnodes(Center, grid, reshape=true)[:, :, 1])
                            .*     cos.(ynodes(Face, grid, reshape=true)[:, :, 1])); # xy slice output

outputs = Dict("scalar" => f, "profile" => g, "slice" => h);

dims = Dict("scalar" => (), "profile" => ("zC",), "slice" => ("xC", "yC"));

output_attributes = Dict(
    "scalar"  => Dict("longname" => "Some scalar", "units" => "bananas"),
    "profile" => Dict("longname" => "Some vertical profile", "units" => "watermelons"),
    "slice"   => Dict("longname" => "Some slice", "units" => "mushrooms")
);

global_attributes = Dict("location" => "Bay of Fundy", "onions" => 7);

simulation.output_writers[:things] =
    NetCDFOutputWriter(model, outputs,
                       schedule=IterationInterval(1), filepath="some_things.nc", dimensions=dims, verbose=true,
                       global_attributes=global_attributes, output_attributes=output_attributes)

# output
NetCDFOutputWriter scheduled on IterationInterval(1):
├── filepath: some_things.nc
├── dimensions: zC(16), zF(17), xC(16), yF(16), xF(16), yC(16), time(0)
├── 3 outputs: ["profile", "slice", "scalar"]
├── field slicer: FieldSlicer(:, :, :, with_halos=false)
└── array type: Array{Float32}
```

See [`NetCDFOutputWriter`](@ref) for more information.

## JLD2 output writer

JLD2 is a fast HDF5 compatible file format written in pure Julia.
JLD2 files can be opened in Julia with the [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) package
and in Python with the [h5py](https://www.h5py.org/) package.

The `JLD2OutputWriter` receives either a `Dict`ionary or `NamedTuple` containing
`name, output` pairs. The `name` can be a symbol or string. The `output` must either be
an `AbstractField` or a function called with `func(model)` that returns arbitrary output.
Whenever output needs to be written, the functions will be called and the output
of the function will be saved to the JLD2 file.

### Examples

Write out 3D fields for u, v, w, and a tracer c, along with a horizontal average:

```jldoctest jld2_output_writer
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

c_avg =  AveragedField(model.tracers.c, dims=(1, 2))

# Note that model.velocities is NamedTuple
simulation.output_writers[:velocities] = JLD2OutputWriter(model, model.velocities,
                                                          prefix = "some_more_data",
                                                          schedule = TimeInterval(20minute),
                                                          init = init_save_some_metadata!)

# output
JLD2OutputWriter scheduled on TimeInterval(20 minutes):
├── filepath: ./some_more_data.jld2
├── 3 outputs: (:u, :v, :w)
├── field slicer: FieldSlicer(:, :, :, with_halos=false)
├── array type: Array{Float32}
├── including: [:grid, :coriolis, :buoyancy, :closure]
└── max filesize: Inf YiB
```

and a time- and horizontal-average of tracer `c` every 20 minutes of simulation time
to a file called `some_more_averaged_data.jld2`

```jldoctest jld2_output_writer
simulation.output_writers[:avg_c] = JLD2OutputWriter(model, (c=c_avg,),
                                                     prefix = "some_more_averaged_data",
                                                     schedule = AveragedTimeInterval(20minute, window=5minute))

# output
JLD2OutputWriter scheduled on TimeInterval(20 minutes):
├── filepath: ./some_more_averaged_data.jld2
├── 1 outputs: (:c,) averaged on AveragedTimeInterval(window=5 minutes, stride=1, interval=20 minutes)
├── field slicer: FieldSlicer(:, :, :, with_halos=false)
├── array type: Array{Float32}
├── including: [:grid, :coriolis, :buoyancy, :closure]
└── max filesize: Inf YiB
```


See [`JLD2OutputWriter`](@ref) for more information.

## Time-averaged output

Time-averaged output is specified by setting the `schedule` keyword argument for either `NetCDFOutputWriter` or
`JLD2OutputWriter` to [`AveragedTimeInterval`](@ref).

With `AveragedTimeInterval`, the time-average of ``a`` is taken as a left Riemann sum corresponding to

```math
\langle a \rangle = \frac{1}{T} \int_{t_i-T}^{t_i} a \, \mathrm{d} t \, ,
```

where ``\langle a \rangle`` is the time-average of ``a``, ``T`` is the time-`window` for averaging specified by
the `window` keyword argument to `AveragedTimeInterval`, and the ``t_i`` are discrete times separated by the
time `interval`. The ``t_i`` specify both the end of the averaging window and the time at which output is written.

### Example

Building an `AveragedTimeInterval` that averages over a 1 year window, every 4 years,

```jldoctest averaged_time_interval
using Oceananigans.OutputWriters: AveragedTimeInterval
using Oceananigans.Utils: year, years

schedule = AveragedTimeInterval(4years, window=1year)

# output
AveragedTimeInterval(window=1 year, stride=1, interval=4 years)
```

An `AveragedTimeInterval` schedule directs an output writer
to time-average its outputs before writing them to disk:

```jldoctest averaged_time_interval
using Oceananigans
using Oceananigans.OutputWriters: JLD2OutputWriter
using Oceananigans.Utils: minutes

model = NonhydrostaticModel(grid=RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)))

simulation = Simulation(model, Δt=10minutes, stop_time=30years)

simulation.output_writers[:velocities] = JLD2OutputWriter(model, model.velocities,
                                                          prefix = "even_more_averaged_velocity_data",
                                                          schedule = AveragedTimeInterval(4years, window=1year, stride=2))

# output
JLD2OutputWriter scheduled on TimeInterval(4 years):
├── filepath: ./even_more_averaged_velocity_data.jld2
├── 3 outputs: (:u, :v, :w) averaged on AveragedTimeInterval(window=1 year, stride=2, interval=4 years)
├── field slicer: FieldSlicer(:, :, :, with_halos=false)
├── array type: Array{Float32}
├── including: [:grid, :coriolis, :buoyancy, :closure]
└── max filesize: Inf YiB
```
