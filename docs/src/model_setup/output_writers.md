# Output writers
Saving model data to disk can be done in a flexible manner using output writers. The two main output writers currently
implemented are a NetCDF output writer (relying on [NCDatasets.jl](https://github.com/Alexander-Barth/NCDatasets.jl))
and a JLD2 output writer (relying on [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)).

Output writers are stored as a list of output writers in `model.output_writers`. Output writers can be specified at
model creation time or be specified at any later time and appended (or assigned with a key value pair) to
`model.output_writers`.

## NetCDF output writer
Model data can be saved to NetCDF files along with associated metadata. The NetCDF output writer is generally used by
passing it a dictionary of (label, field) pairs and any indices for slicing if you don't want to save the full 3D field.

The following example shows how to construct NetCDF output writers for two different kinds of outputs (3D fields and
slices) along with output attributes
```@example
Nx = Ny = Nz = 16
model = Model(grid=RegularCartesianGrid(size=(Nx, Ny, Nz), length=(1, 1, 1)))

fields = Dict(
    "u" => model.velocities.u,
    "T" => model.tracers.T
)

output_attributes = Dict(
    "u" => Dict("longname" => "Velocity in the x-direction", "units" => "m/s"),
    "T" => Dict("longname" => "Temperature", "units" => "C")
)

model.output_writers[:field_writer] = NetCDFOutputWriter(model, fields; filename="output_fields.nc",
                                                         interval=6hour, output_attributes=output_attributes)

model.output_writers[:surface_slice_writer] = NetCDFOutputWriter(model, fields; filename="output_surface_xy_slice.nc",
                                                                 interval=5minute, output_attributes=output_attributes,
                                                                 zC=Nz, zF=Nz)
```

See [`NetCDFOutputWriter`](@ref) for more details and options.

## JLD2 output writer
JLD2 is a an HDF5 compatible file format written in pure Julia and is generally pretty fast. JLD2 files can be opened in
Python with the [h5py](https://www.h5py.org/) package.

The JLD2 output writer is generally used by passing it a dictionary or named tuple of (label, function) pairs where the
functions have a single input `model`. Whenever output needs to be written, the functions will be called and the output
of the function will be saved to the JLD2 file. For example, to write out 3D fields for w and T and a horizontal average
of T every 1 hour of simulation time to a file called `some_data.jld2`
```@example
model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)))

function init_save_some_metadata(file, model)
    file["author"] = "Chim Riggles"
    file["parameters/coriolis_parameter"] = 1e-4
    file["parameters/density"] = 1027
end

T_avg =  HorizontalAverage(model.tracers.T)

outputs = Dict(
    :w => model -> model.velocities.u,
    :T => model -> model.tracers.T,
    :T_avg => model -> T_avg(model)
)

jld2_writer = JLD2OutputWriter(model, outputs; init=init_save_some_metadata, interval=1hour, prefix="some_data")

push!(model.output_writers, jld2_writer)
```

See [`JLD2OutputWriter`](@ref) for more details and options.

## Checkpointer
A checkpointer can be used to serialize the entire model state to a file from which the model can be restored at any
time. This is useful if you'd like to periodically checkpoint when running long simulations in case of crashes or
cluster time limits, but also if you'd like to restore from a checkpoint and try out multiple scenarios.

For example, to periodically checkpoint the model state to disk every 1,000,000 seconds of simulation time to files of
the form `model_checkpoint_xxx.jld2` where `xxx` is the iteration number (automatically filled in)
```@example
model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)))
model.output_writers[:checkpointer] = Checkpointer(model; interval=1e6, prefix="model_checkpoint")
```

The default options should provide checkpoint files that are easy to restore from in most cases. For more advanced
options and features, see [`Checkpointer`](@ref).

### Restoring from a checkpoint file
To restore the model from a checkpoint file, for example `model_checkpoint_12345.jld2`, simply call
```
model = restore_from_checkpoint("model_checkpoint_12345.jld2")
```
which will create a new model object that is identical to the one that was serialized to disk. You can continue time
stepping after restoring from a checkpoint.

You can pass additional parameters to the `Model` constructor. See [`restore_from_checkpoint`](@ref) for more
information.

### Restoring with functions
JLD2 cannot serialize functions to disk. so if you used forcing functions, boundary conditions containing functions, or
the model included references to functions then they will not be serialized to the checkpoint file. When restoring from
a checkpoint file, any model property that contained functions must be manually restored via keyword arguments to
[`restore_from_checkpoint`](@ref).
