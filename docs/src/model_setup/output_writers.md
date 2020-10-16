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

```jldoctest netcdf1
using Oceananigans, Oceananigans.OutputWriters

grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1));

model = IncompressibleModel(grid=grid);

simulation = Simulation(model, Δt=12, stop_time=3600);

fields = Dict("u" => model.velocities.u, "T" => model.tracers.T);

simulation.output_writers[:field_writer] =
    NetCDFOutputWriter(model, fields, filepath="output_fields.nc", time_interval=60)

# output
NetCDFOutputWriter (time_interval=60): output_fields.nc
├── dimensions: zC(16), zF(17), xC(16), yF(16), xF(16), yC(16), time(0)
└── 2 outputs: ["T", "u"]
```

```jldoctest netcdf1
simulation.output_writers[:surface_slice_writer] =
    NetCDFOutputWriter(model, fields, filepath="output_surface_xy_slice.nc",
                       time_interval=60, field_slicer=FieldSlicer(k=grid.Nz))

# output
NetCDFOutputWriter (time_interval=60): output_surface_xy_slice.nc
├── dimensions: zC(1), zF(1), xC(16), yF(16), xF(16), yC(16), time(0)
└── 2 outputs: ["T", "u"]
```

Writing a scalar, profile, and slice to NetCDF:

```jldoctest
using Oceananigans, Oceananigans.OutputWriters

grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 2, 3));

model = IncompressibleModel(grid=grid);

simulation = Simulation(model, Δt=1.25, stop_iteration=3);

f(model) = model.clock.time^2; # scalar output

g(model) = model.clock.time .* exp.(znodes(Cell, grid)); # vector/profile output

h(model) = model.clock.time .* (   sin.(xnodes(Cell, grid, reshape=true)[:, :, 1])
                            .*     cos.(ynodes(Face, grid, reshape=true)[:, :, 1])); # xy slice output

outputs = Dict("scalar" => f, "profile" => g, "slice" => h);

dims = Dict("scalar" => (), "profile" => ("zC",), "slice" => ("xC", "yC"));

output_attributes = Dict(
    "scalar"  => Dict("longname" => "Some scalar", "units" => "bananas"),
    "profile" => Dict("longname" => "Some vertical profile", "units" => "watermelons"),
    "slice"   => Dict("longname" => "Some slice", "units" => "mushrooms")
);

global_attributes = Dict("location" => "Bay of Fundy", "onions" => 7);

simulation.output_writers[:stuff] =
    NetCDFOutputWriter(model, outputs,
                       iteration_interval=1, filepath="stuff.nc", dimensions=dims, verbose=true,
                       global_attributes=global_attributes, output_attributes=output_attributes)

# output
NetCDFOutputWriter (iteration_interval=1): stuff.nc
├── dimensions: zC(16), zF(17), xC(16), yF(16), xF(16), yC(16), time(0)
└── 3 outputs: ["profile", "slice", "scalar"]
```

See [`NetCDFOutputWriter`](@ref) for more details and options.

## JLD2 output writer

JLD2 is a an HDF5 compatible file format written in pure Julia and is generally pretty fast. JLD2 files can be opened in
Python with the [h5py](https://www.h5py.org/) package.

The JLD2 output writer is generally used by passing it a dictionary or named tuple of (label, function) pairs where the
functions have a single input `model`. Whenever output needs to be written, the functions will be called and the output
of the function will be saved to the JLD2 file. For example, to write out 3D fields for w and T and a horizontal average
of T every 1 hour of simulation time to a file called `some_data.jld2`

```julia
using Oceananigans
using Oceananigans.OutputWriters
using Oceananigans.Utils: hour, minute

model = IncompressibleModel(grid=RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1)))
simulation = Simulation(model, Δt=12, stop_time=1hour)

function init_save_some_metadata(file, model)
    file["author"] = "Chim Riggles"
    file["parameters/coriolis_parameter"] = 1e-4
    file["parameters/density"] = 1027
end

T_avg =  AveragedField(model.tracers.T, dims=(1, 2))

outputs = Dict(
    :w => model -> model.velocities.u,
    :T => model -> model.tracers.T,
    :T_avg => model -> T_avg(model)
)

jld2_writer = JLD2OutputWriter(model, outputs, init=init_save_some_metadata, interval=20minute, prefix="some_data")

push!(simulation.output_writers, jld2_writer)
```

See [`JLD2OutputWriter`](@ref) for more details and options.
