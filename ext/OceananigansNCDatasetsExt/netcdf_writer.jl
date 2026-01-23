#####
##### NetCDF Output Writer for Oceananigans
#####
#
# This file implements NetCDFWriter, which saves Oceananigans simulation output to
# NetCDF files during simulation runtime. By default, it also saves grid reconstruction
# data and immersed boundary construction parameters to the NetCDF file, as well as
# grid metrics.
#

#####
##### Extend defVar to be able to write fields to NetCDF directly
#####

defVar(ds::AbstractDataset, name, op::AbstractOperation; kwargs...) = defVar(ds, name, Field(op); kwargs...)
defVar(ds::AbstractDataset, name, op::Reduction; kwargs...) = defVar(ds, name, Field(op); kwargs...)

function defVar(ds::AbstractDataset, field_name, fd::AbstractField;
                array_type=Array{eltype(fd)},
                time_dependent=false,
                with_halos=false,
                dimension_name_generator = trilocation_dim_name,
                dimension_type=Float64,
                write_data=true,
                kwargs...)

    # effective_dim_names are the dimensions that will be used to write the field data (excludes reduced and dimensions where location is Nothing)
    effective_dim_names = create_field_dimensions!(ds, fd, dimension_name_generator; time_dependent, with_halos, array_type, dimension_type)

    # Add location to attributes
    loc = location(fd) |> convert_for_netcdf
    loc_attrib = Dict("location" => loc)
    if :attrib ∈ keys(kwargs)
        attrib = merge(loc_attrib, kwargs[:attrib])
    else
        attrib = loc_attrib
    end

    # Add indices to attributes
    attrib = merge(attrib, Dict("indices" => convert_for_netcdf(indices(fd))))
    kwargs = merge(kwargs, pairs((; attrib,)))

    # Write the data to the NetCDF file (or don't, but still create the space for it there)
    if write_data
        # Squeeze the data to remove dimensions where location is Nothing and add a time dimension if the field is time-dependent
        constructed_fd = construct_output(fd, (:, :, :), with_halos)
        squeezed_field_data = squeeze_nothing_dimensions(constructed_fd; array_type)
        squeezed_reshaped_field_data = time_dependent ? reshape(squeezed_field_data, size(squeezed_field_data)..., 1) : squeezed_field_data

        defVar(ds, field_name, squeezed_reshaped_field_data, effective_dim_names; kwargs...)
    else
        defVar(ds, field_name, eltype(array_type), effective_dim_names; kwargs...)
    end
end

defVar(ds::AbstractDataset, field_name::Union{AbstractString, Symbol}, data::Array{Bool}, dim_names; kwargs...) = defVar(ds, field_name, Int8.(data), dim_names; kwargs...)

#####
##### Variable attributes
#####

default_velocity_attributes(::RectilinearGrid) = Dict(
    "u" => Dict("long_name" => "Velocity in the +x-direction.", "units" => "m/s"),
    "v" => Dict("long_name" => "Velocity in the +y-direction.", "units" => "m/s"),
    "w" => Dict("long_name" => "Velocity in the +z-direction.", "units" => "m/s"))

default_velocity_attributes(::LatitudeLongitudeGrid) = Dict(
    "u"            => Dict("long_name" => "Velocity in the zonal direction (+ = east).", "units" => "m/s"),
    "v"            => Dict("long_name" => "Velocity in the meridional direction (+ = north).", "units" => "m/s"),
    "w"            => Dict("long_name" => "Velocity in the vertical direction (+ = up).", "units" => "m/s"),
    "displacement" => Dict("long_name" => "Sea surface height displacement", "units" => "m"))

default_velocity_attributes(ibg::ImmersedBoundaryGrid) = default_velocity_attributes(ibg.underlying_grid)

default_tracer_attributes(::Nothing) = Dict()

default_tracer_attributes(::BuoyancyForce{<:BuoyancyTracer}) = Dict("b" => Dict("long_name" => "Buoyancy", "units" => "m/s²"))

default_tracer_attributes(::BuoyancyForce{<:SeawaterBuoyancy{FT, <:LinearEquationOfState}}) where FT = Dict(
    "T" => Dict("long_name" => "Temperature", "units" => "°C"),
    "S" => Dict("long_name" => "Salinity",    "units" => "practical salinity unit (psu)"))

default_tracer_attributes(::BuoyancyBoussinesqEOSModel) = Dict("T" => Dict("long_name" => "Conservative temperature", "units" => "°C"),
                                                               "S" => Dict("long_name" => "Absolute salinity",        "units" => "g/kg"))

function default_output_attributes(model)
    velocity_attrs = default_velocity_attributes(model.grid)
    buoyancy = model isa ShallowWaterModel ? nothing : model.buoyancy
    tracer_attrs = default_tracer_attributes(buoyancy)
    return merge(velocity_attrs, tracer_attrs)
end

#####
##### Saving schedule metadata as global attributes
#####

add_schedule_metadata!(attributes, schedule) = nothing

function add_schedule_metadata!(global_attributes, schedule::IterationInterval)
    global_attributes["schedule"] = "IterationInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output iteration interval"] = "Output was saved every $(schedule.interval) iteration(s)."

    return nothing
end

function add_schedule_metadata!(global_attributes, schedule::TimeInterval)
    global_attributes["schedule"] = "TimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] = "Output was saved every $(prettytime(schedule.interval))."

    return nothing
end

function add_schedule_metadata!(global_attributes, schedule::WallTimeInterval)
    global_attributes["schedule"] = "WallTimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] =
        "Output was saved every $(prettytime(schedule.interval))."

    return nothing
end

function add_schedule_metadata!(global_attributes, schedule::AveragedTimeInterval)
    global_attributes["schedule"] = "AveragedTimeInterval"
    global_attributes["interval"] = schedule.interval
    global_attributes["output time interval"] = "Output was time-averaged and saved every $(prettytime(schedule.interval))."

    global_attributes["time_averaging_window"] = schedule.window
    global_attributes["time averaging window"] = "Output was time averaged with a window size of $(prettytime(schedule.window))"

    global_attributes["time_averaging_stride"] = schedule.stride
    global_attributes["time averaging stride"] = "Output was time averaged with a stride of $(schedule.stride) iteration(s) within the time averaging window."

    return nothing
end

#####
##### NetCDFWriter constructor
#####

"""
    NetCDFWriter(model::AbstractModel, outputs;
                 filename,
                 schedule,
                 grid = model.grid,
                 dir = ".",
                 array_type = Array{Float32},
                 indices = (:, :, :),
                 global_attributes = Dict(),
                 output_attributes = Dict(),
                 dimensions = Dict(),
                 with_halos = false,
                 include_grid_metrics = true,
                 overwrite_existing = nothing,
                 verbose = false,
                 deflatelevel = 0,
                 part = 1,
                 file_splitting = NoFileSplitting(),
                 dimension_name_generator = trilocation_dim_name,
                 dimension_type = Float64)

Construct a `NetCDFWriter` that writes `(label, output)` pairs in `outputs` to a NetCDF file.

!!! note "NCDatasets required"
    `NetCDFWriter` requires NCDatasets.jl to be loaded: `using NCDatasets`

The `outputs` can be a `Dict` or `NamedTuple` where each `label` is a symbol or string and each `output` is one of:

- An `AbstractField` (e.g., `model.velocities.u`, `model.tracers.T`)
- An `AbstractOperation` or `Reduction` (e.g., `Average(model.tracers.T, dims=(1, 2))`)
- `LagrangianParticles` for particle tracking data
- A function `f(model)` that returns data to be written to disk

If any `outputs` are not `AbstractField`, `AbstractOperation`, `Reduction`, or `LagrangianParticles`,
their spatial `dimensions` must be provided as a `Dict` or `NamedTuple` mapping output names to
dimension name tuples.

Required arguments
==================

- `model`: The Oceananigans model instance.

- `outputs`: A collection of outputs to write, specified as either:
  * A `Dict` with `Symbol` or `String` keys and field/operation/function values
  * A `NamedTuple` of fields, operations, or functions

Required keyword arguments
==========================

- `filename`: Descriptive filename. `".nc"` is appended automatically if not present.

- `schedule`: An `AbstractSchedule` that determines when output is saved. Options include:
  * `TimeInterval(dt)`: Save every `dt` seconds of simulation time
  * `IterationInterval(n)`: Save every `n` iterations
  * `AveragedTimeInterval(dt; window, stride)`: Time-average output over a window before saving
  * `WallTimeInterval(dt)`: Save every `dt` seconds of wall clock time

Optional keyword arguments
==========================

- `grid`: The grid associated with `outputs`. Default: `model.grid`.
          Use this to specify a different grid when outputs are interpolated or regridded.

- `dir`: Directory to save output to. Default: `"."`.

- `array_type`: Type to convert outputs to before saving. Default: `Array{Float32}`.

- `indices`: Tuple of indices of the output variables to include. Default is `(:, :, :)`, which
             includes the full fields. This allows saving specific slices of the domain.

- `global_attributes`: `Dict` or `NamedTuple` of global attributes or metadata to save with every file.
                       Default: `Dict()`. This is useful for saving information specific to the simulation.
                       Some useful global attributes are included by default but will be overwritten if
                       included in this `Dict`.

- `output_attributes`: `Dict` or `NamedTuple` of attributes to save with each output variable.
                       Default: `Dict()`.
                       Reasonable defaults (long_name, units) are provided for standard variables
                       (u, v, w, T, S, b) and can be overwritten here.

- `dimensions`: A `Dict` or `NamedTuple` of dimension tuples to apply to outputs (required for function
                outputs that return custom data).

- `with_halos`: Boolean defining whether to include halos in the outputs. Default: `false`.
                Note that to postprocess saved output (e.g., compute derivatives, etc.),
                information about the boundary conditions is often crucial. In those cases,
                you might need to set `with_halos = true`. Cannot be used with custom `indices`.

- `include_grid_metrics`: Include grid metrics such as grid spacings, areas, and volumes as
                          additional variables. Default: `true`. Note that even with
                          `include_grid_metrics = false`, core grid coordinates are still saved.

- `overwrite_existing`: If `false`, `NetCDFWriter` will append to existing files. If `true`,
                        it will overwrite existing files or create new ones. Default: `true` if the
                        file does not exist, `false` if it does.

- `verbose`: Log variable compute times, file write times, and file sizes. Default: `false`.

- `deflatelevel`: Determines the NetCDF compression level of data (integer 0-9; 0 (default) means no compression
                  and 9 means maximum compression). See [NCDatasets.jl documentation](https://alexander-barth.github.io/NCDatasets.jl/stable/variables/#Creating-a-variable)
                  for more information.

- `part`: Starting part number for file splitting. Default: `1`.

- `file_splitting`: Schedule for splitting the output file. The new files will be suffixed with
                    `_part1`, `_part2`, etc. Options include:
                    * `FileSizeLimit(sz)`: Split when file size exceeds `sz` (e.g., `200KiB`).
                    * `TimeInterval(interval)`: Split every `interval` of simulation time.
                    * `NoFileSplitting()` (default): Don't split files.

- `dimension_name_generator`: A function with signature `(var_name, grid, LX, LY, LZ, dim)` where `dim` is
                              either `Val(:x)`, `Val(:y)`, or `Val(:z)` that returns a string corresponding
                              to the name of the dimension `var_name` on `grid` with location `(LX, LY, LZ)`
                              along `dim`. This advanced option can be used to rename dimensions and variables
                              to satisfy certain naming conventions. Default: `trilocation_dim_name`.

- `dimension_type`: Floating point type for dimension coordinate arrays. Default: `Float64`.
                    Use `Float32` to reduce file size if needed.

Examples
========

Saving the ``u`` velocity field and temperature fields, the full 3D fields and surface 2D slices
to separate NetCDF files:

```@example netcdf1
using Oceananigans

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1))

model = NonhydrostaticModel(grid=grid, tracers=:c)

simulation = Simulation(model, Δt=12, stop_time=3600)

fields = Dict("u" => model.velocities.u, "c" => model.tracers.c)

simulation.output_writers[:field_writer] =
    NetCDFWriter(model, fields, filename="fields.nc", schedule=TimeInterval(60))
```

```@example netcdf1
simulation.output_writers[:surface_slice_writer] =
    NetCDFWriter(model, fields, filename="surface_xy_slice.nc",
                 schedule=TimeInterval(60), indices=(:, :, grid.Nz))
```

```@example netcdf1
simulation.output_writers[:averaged_profile_writer] =
    NetCDFWriter(model, fields,
                 filename = "averaged_z_profile.nc",
                 schedule = AveragedTimeInterval(60, window=20),
                 indices = (1, 1, :))
```

`NetCDFWriter` also accepts output functions that write scalars and arrays to disk,
provided that their `dimensions` are provided:

```@example
using Oceananigans

Nx, Ny, Nz = 16, 16, 16

grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(1, 2, 3))

model = NonhydrostaticModel(; grid)

simulation = Simulation(model, Δt=1.25, stop_iteration=3)

f(model) = model.clock.time^2 # scalar output

zC = znodes(grid, Center())
g(model) = model.clock.time .* exp.(zC) # vector/profile output

xC, yF = xnodes(grid, Center()), ynodes(grid, Face())
XC = [xC[i] for i in 1:Nx, j in 1:Ny]
YF = [yF[j] for i in 1:Nx, j in 1:Ny]
h(model) = @. model.clock.time * sin(XC) * cos(YF) # xy slice output

outputs = Dict("scalar" => f, "profile" => g, "slice" => h)

dims = Dict("scalar" => (), "profile" => ("zC",), "slice" => ("xC", "yC"))

output_attributes = Dict(
    "scalar"  => Dict("long_name" => "Some scalar", "units" => "bananas"),
    "profile" => Dict("long_name" => "Some vertical profile", "units" => "watermelons"),
    "slice"   => Dict("long_name" => "Some slice", "units" => "mushrooms"))

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
"""
function NetCDFWriter(model::AbstractModel, outputs;
                      filename,
                      schedule,
                      grid = model.grid,
                      dir = ".",
                      array_type = Array{Float32},
                      indices = (:, :, :),
                      global_attributes = Dict(),
                      output_attributes = Dict(),
                      dimensions = Dict(),
                      with_halos = false,
                      include_grid_metrics = true,
                      overwrite_existing = nothing,
                      verbose = false,
                      deflatelevel = 0,
                      part = 1,
                      file_splitting = NoFileSplitting(),
                      dimension_name_generator = trilocation_dim_name,
                      dimension_type = Float64)

    if with_halos && indices != (:, :, :)
        throw(ArgumentError("If with_halos=true then you cannot pass indices: $indices"))
    end

    mkpath(dir)
    filename = auto_extension(filename, ".nc")
    filepath = abspath(joinpath(dir, filename))

    initialize!(file_splitting, model)

    schedule = materialize_schedule(schedule)
    update_file_splitting_schedule!(file_splitting, filepath)

    if isnothing(overwrite_existing)
        if isfile(filepath)
            overwrite_existing = false
        else
            overwrite_existing = true
        end
    else
        if isfile(filepath) && !overwrite_existing
            @warn "$filepath already exists and `overwrite_existing = false`. Mode will be set to append to existing file. " *
                  "You might experience errors when writing output if the existing file belonged to a different simulation!"

        elseif isfile(filepath) && overwrite_existing
            @warn "Overwriting existing $filepath."
        end
    end

    outputs = Dict(string(name) => construct_output(outputs[name], indices, with_halos) for name in keys(outputs))

    output_attributes = dictify(output_attributes)
    global_attributes = dictify(global_attributes)
    dimensions = dictify(dimensions)

    # Ensure we can add any kind of metadata to the attributes later by converting to Dict{Any, Any}.
    global_attributes = Dict{Any, Any}(global_attributes)

    dataset, outputs, schedule = initialize_nc_file(model,
                                                    grid,
                                                    filepath,
                                                    outputs,
                                                    schedule,
                                                    array_type,
                                                    indices,
                                                    global_attributes,
                                                    output_attributes,
                                                    dimensions,
                                                    with_halos,
                                                    include_grid_metrics,
                                                    overwrite_existing,
                                                    deflatelevel,
                                                    dimension_name_generator,
                                                    dimension_type)

    return NetCDFWriter(grid,
                        filepath,
                        dataset,
                        outputs,
                        schedule,
                        array_type,
                        indices,
                        global_attributes,
                        output_attributes,
                        dimensions,
                        with_halos,
                        include_grid_metrics,
                        overwrite_existing,
                        verbose,
                        deflatelevel,
                        part,
                        file_splitting,
                        dimension_name_generator,
                        dimension_type)
end

#####
##### NetCDF file initialization
#####

function initialize_nc_file(model,
                            grid,
                            filepath,
                            outputs,
                            schedule,
                            array_type,
                            indices,
                            global_attributes,
                            output_attributes,
                            dimensions,
                            with_halos,
                            include_grid_metrics,
                            overwrite_existing,
                            deflatelevel,
                            dimension_name_generator,
                            dimension_type)

    mode = overwrite_existing ? "c" : "a"

    # Add useful metadata
    useful_attributes = Dict("date" => "This file was generated on $(now()) local time ($(now(UTC)) UTC).",
                             "Julia" => "This file was generated using " * versioninfo_with_gpu(),
                             "Oceananigans" => "This file was generated using " * oceananigans_versioninfo())

    if with_halos
        useful_attributes["output_includes_halos"] =
            "The outputs include data from the halo regions of the grid."
    end

    global_attributes = merge(useful_attributes, global_attributes)

    add_schedule_metadata!(global_attributes, schedule)

    # Convert schedule to TimeInterval and each output to WindowedTimeAverage if
    # schedule::AveragedTimeInterval
    schedule, outputs = time_average_outputs(schedule, outputs, model)

    dims = gather_dimensions(outputs, grid, indices, with_halos, dimension_name_generator)

    # Open the NetCDF dataset file
    dataset = NCDataset(filepath, mode, attrib=sort(collect(pairs(global_attributes)), by=first))

    # Merge the default with any user-supplied output attributes, ensuring the user-supplied ones
    # can overwrite the defaults.
    output_attributes = merge(default_dimension_attributes(grid, dimension_name_generator),
                              default_output_attributes(model),
                              output_attributes)

    # Define variables for each dimension and attributes if this is a new file.
    if mode == "c"
        # This metadata is to support `FieldTimeSeries`.
        write_grid_reconstruction_data!(dataset, grid; array_type, deflatelevel)

        # DateTime and TimeDate are both <: AbstractTime
        time_attrib = model.clock.time isa AbstractTime ?
            Dict("long_name" => "Time", "units" => "seconds since 2000-01-01 00:00:00") :
            Dict("long_name" => "Time", "units" => "seconds")

        create_time_dimension!(dataset, attrib=time_attrib, dimension_type=dimension_type)
        create_spatial_dimensions!(dataset, dims, output_attributes; deflatelevel=1, dimension_type=dimension_type)

        time_independent_vars = Dict()

        if include_grid_metrics
            grid_metrics = gather_grid_metrics(grid, indices, dimension_name_generator)
            merge!(time_independent_vars, grid_metrics)
        end

        if grid isa ImmersedBoundaryGrid
            immersed_boundary_vars = gather_immersed_boundary(grid, indices, dimension_name_generator)
            merge!(time_independent_vars, immersed_boundary_vars)
        end

        if !isempty(time_independent_vars)
            for (output_name, output) in sort(collect(pairs(time_independent_vars)), by=first)
                output = construct_output(output, indices, with_halos)
                attrib = haskey(output_attributes, output_name) ? output_attributes[output_name] : Dict()
                materialized = materialize_output(output, model)

                define_output_variable!(model,
                                        dataset,
                                        materialized,
                                        output_name;
                                        array_type,
                                        deflatelevel,
                                        attrib,
                                        dimensions,
                                        filepath, # for better error messages
                                        dimension_name_generator,
                                        time_dependent = false,
                                        with_halos,
                                        dimension_type)

                save_output!(dataset, output, model, output_name, array_type)
            end
        end

        for (output_name, output) in sort(collect(pairs(outputs)), by=first)
            attrib = haskey(output_attributes, output_name) ? output_attributes[output_name] : Dict()
            materialized = materialize_output(output, model)

            define_output_variable!(model,
                                    dataset,
                                    materialized,
                                    output_name;
                                    array_type,
                                    deflatelevel,
                                    attrib,
                                    dimensions,
                                    filepath, # for better error messages
                                    dimension_name_generator,
                                    time_dependent = true,
                                    with_halos,
                                    dimension_type)
        end

        sync(dataset)
    end

    close(dataset)

    return dataset, outputs, schedule
end

initialize_nc_file(ow::NetCDFWriter, model) = initialize_nc_file(model,
                                                                 ow.grid,
                                                                 ow.filepath,
                                                                 ow.outputs,
                                                                 ow.schedule,
                                                                 ow.array_type,
                                                                 ow.indices,
                                                                 ow.global_attributes,
                                                                 ow.output_attributes,
                                                                 ow.dimensions,
                                                                 ow.with_halos,
                                                                 ow.include_grid_metrics,
                                                                 ow.overwrite_existing,
                                                                 ow.deflatelevel,
                                                                 ow.dimension_name_generator,
                                                                 ow.dimension_type)

#####
##### Variable definition
#####

materialize_output(func, model) = func(model)
materialize_output(field::AbstractField, model) = field
materialize_output(particles::LagrangianParticles, model) = particles
materialize_output(output::WindowedTimeAverage{<:AbstractField}, model) = output

""" Defines empty variables for 'custom' user-supplied `output`. """
function define_output_variable!(model, dataset, output, output_name; array_type,
                                 deflatelevel, attrib, dimension_name_generator,
                                 time_dependent, with_halos,
                                 dimensions, filepath, dimension_type=Float64)

    if output_name ∉ keys(dimensions)
        msg = string("dimensions[$output_name] for output $output_name=$(typeof(output)) into $filepath" *
                     " must be provided when constructing NetCDFWriter")
        throw(ArgumentError(msg))
    end

    dims = dimensions[output_name]
    FT = eltype(array_type)
    all_dims = time_dependent ? (dims..., "time") : dims
    defVar(dataset, output_name, FT, all_dims; deflatelevel, attrib)

    return nothing
end

""" Defines empty field variable. """
function define_output_variable!(model, dataset, output::AbstractField, output_name; array_type,
                                 deflatelevel, attrib, dimension_name_generator,
                                 time_dependent, with_halos,
                                 dimensions, filepath, dimension_type=Float64)

    # If the output is the free surface, we need to handle it differently since it will be writen as a 3D array with a singleton dimension for the z-coordinate
    if output_name == "displacement" && hasfield(typeof(model), :free_surface)
        if output == view(model.free_surface.displacement, output.indices...)
            local default_dimension_name_generator = dimension_name_generator
            dimension_name_generator = (var_name, grid, LX, LY, LZ, dim) -> dimension_name_generator_free_surface(default_dimension_name_generator, var_name, grid, LX, LY, LZ, dim)
        end
    end
    defVar(dataset, output_name, output; array_type, time_dependent, with_halos, dimension_name_generator, deflatelevel, attrib, dimension_type, write_data=false)
    return nothing
end

""" Defines empty field variable for `WindowedTimeAverage`s over fields. """
define_output_variable!(model, dataset, output::WindowedTimeAverage{<:AbstractField}, output_name; kwargs...) =
    define_output_variable!(model, dataset, output.operand, output_name; kwargs...)

""" Defines empty variable for particle trackting. """
function define_output_variable!(model, dataset, output::LagrangianParticles, output_name; array_type,
                                 deflatelevel, kwargs...)

    particle_fields = eltype(output.properties) |> fieldnames .|> string
    T = eltype(array_type)

    for particle_field in particle_fields
        defVar(dataset, particle_field, T, ("particle_id", "time"); deflatelevel)
    end

    return nothing
end

#####
##### Write output
#####

Base.open(nc::NetCDFWriter) = NCDataset(nc.filepath, "a")
Base.close(nc::NetCDFWriter) = close(nc.dataset)

# Saving outputs with no time dependence (e.g. grid metrics)
function save_output!(ds, output, model, output_name, array_type)
    fetched = fetch_output(output, model)
    data = convert_output(fetched, array_type)
    data = squeeze_nothing_dimensions(output, data)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    ds[output_name][colons...] = data
    return nothing
end

# Saving time-dependent outputs
function save_output!(ds, output, model, ow, time_index, output_name)
    data = fetch_and_convert_output(output, model, ow)
    data = squeeze_nothing_dimensions(output, data)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    ds[output_name][colons..., time_index:time_index] = data
    return nothing
end

function save_output!(ds, output::LagrangianParticles, model, ow, time_index, name)
    data = fetch_and_convert_output(output, model, ow)
    for (particle_field, vals) in pairs(data)
        ds[string(particle_field)][:, time_index] = vals
    end

    return nothing
end

# Convert to a base Julia type (a float or DateTime).
float_or_date_time(t) = t
float_or_date_time(t::AbstractTime) = DateTime(t)

"""
    write_output!(ow::NetCDFWriter, model)

Write output to netcdf file `output_writer.filepath` at specified intervals. Increments the `time` dimension
every time an output is written to the file.
"""
function write_output!(ow::NetCDFWriter, model::AbstractModel)
    # Start a new file if the file_splitting(model) is true
    ow.file_splitting(model) && start_next_file(model, ow)
    update_file_splitting_schedule!(ow.file_splitting, ow.filepath)

    ow.dataset = open(ow)

    ds, verbose, filepath = ow.dataset, ow.verbose, ow.filepath

    time_index = length(ds["time"]) + 1
    ds["time"][time_index] = float_or_date_time(model.clock.time)

    if verbose
        @info "Writing to NetCDF: $filepath..."
        @info "Computing NetCDF outputs for time index $(time_index): $(keys(ow.outputs))..."

        # Time and file size before computing any outputs.
        t0, sz0 = time_ns(), filesize(filepath)
    end

    for (output_name, output) in ow.outputs
        # Time before computing this output.
        verbose && (t0′ = time_ns())

        save_output!(ds, output, model, ow, time_index, output_name)

        if verbose
            # Time after computing this output.
            t1′ = time_ns()
            @info "Computing $output_name done: time=$(prettytime((t1′-t0′) / 1e9))"
        end
    end

    sync(ds)
    close(ow)

    if verbose
        # Time and file size after computing and writing all outputs to disk.
        t1, sz1 = time_ns(), filesize(filepath)
        verbose && @info begin
            @sprintf("Writing done: time=%s, size=%s, Δsize=%s",
                    prettytime((t1-t0)/1e9), pretty_filesize(sz1), pretty_filesize(sz1-sz0))
        end
    end

    return nothing
end

#####
##### Show
#####

Base.summary(ow::NetCDFWriter) =
    string("NetCDFWriter writing ", prettykeys(ow.outputs), " to ", ow.filepath, " on ", summary(ow.schedule))

function Base.show(io::IO, ow::NetCDFWriter)
    dims = NCDataset(ow.filepath, "r") do ds
        join([dim * "(" * string(length(ds[dim])) * "), "
              for dim in keys(ds.dim)])[1:end-2]
    end

    averaging_schedule = output_averaging_schedule(ow)
    num_outputs = length(ow.outputs)

    print(io, "NetCDFWriter scheduled on $(summary(ow.schedule)):", "\n",
              "├── filepath: ", relpath(ow.filepath), "\n",
              "├── dimensions: $dims", "\n",
              "├── $num_outputs outputs: ", prettykeys(ow.outputs), show_averaging_schedule(averaging_schedule), "\n",
              "├── array_type: ", show_array_type(ow.array_type), "\n",
              "├── file_splitting: ", summary(ow.file_splitting), "\n",
              "└── file size: ", pretty_filesize(filesize(ow.filepath)))
end

#####
##### File splitting
#####

function start_next_file(model, ow::NetCDFWriter)
    verbose = ow.verbose

    verbose && @info begin
        schedule_type = summary(ow.file_splitting)
        "Splitting output because $(schedule_type) is activated."
    end

    if ow.part == 1
        part1_path = replace(ow.filepath, r".nc$" => "_part1.nc")
        verbose && @info "Renaming first part: $(ow.filepath) -> $part1_path"
        mv(ow.filepath, part1_path, force=ow.overwrite_existing)
        ow.filepath = part1_path
    end

    ow.part += 1
    ow.filepath = replace(ow.filepath, r"part\d+.nc$" => "part" * string(ow.part) * ".nc")
    ow.overwrite_existing && isfile(ow.filepath) && rm(ow.filepath, force=true)
    verbose && @info "Now writing to: $(ow.filepath)"

    initialize_nc_file(ow, model)

    return nothing
end
