# Output writers
Saving model data to disk can be done in a flexible manner using output writers. The two main output writers currently
implemented are a NetCDF output writer (relying on [NCDatasets.jl](https://github.com/Alexander-Barth/NCDatasets.jl))
and a JLD2 output writer (relying on [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)).

Output writers are stored as a list of output writers in `simulation.output_writers`. Output writers can be specified
at model creation time or be specified at any later time and appended (or assigned with a key value pair) to
`simulation.output_writers`.

## NetCDF output writer
Model data can be saved to NetCDF files along with associated metadata. The NetCDF output writer is generally used by
passing it a dictionary of (label, field) pairs and any indices for slicing if you don't want to save the full 3D field.

The following example shows how to construct NetCDF output writers for two different kinds of outputs (3D fields and
slices) along with output attributes
```@example
Nx = Ny = Nz = 16
model = Model(grid=RegularCartesianGrid(size=(Nx, Ny, Nz), length=(1, 1, 1)))
simulation = Simulation(model, Δt=12, stop_time=3600)

fields = Dict(
    "u" => model.velocities.u,
    "T" => model.tracers.T
)

output_attributes = Dict(
    "u" => Dict("longname" => "Velocity in the x-direction", "units" => "m/s"),
    "T" => Dict("longname" => "Temperature", "units" => "C")
)

simulation.output_writers[:field_writer] =
    NetCDFOutputWriter(model, fields; filename="output_fields.nc",
                       interval=6hour, output_attributes=output_attributes)

simulation.output_writers[:surface_slice_writer] =
    NetCDFOutputWriter(model, fields; filename="output_surface_xy_slice.nc",
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
simulation = Simulation(model, Δt=12, stop_time=3600)

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

jld2_writer = JLD2OutputWriter(model, outputs, init=init_save_some_metadata, interval=1hour, prefix="some_data")

push!(simulation.output_writers, jld2_writer)
```

See [`JLD2OutputWriter`](@ref) for more details and options.
