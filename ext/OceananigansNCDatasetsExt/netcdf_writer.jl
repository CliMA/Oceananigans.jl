#####
##### NetCDF Output Writer for Oceananigans
#####
#
# This file implements NetCDFWriter, which saves Oceananigans simulation output to
# NetCDF files during simulation runtime. By default, it also saves grid reconstruction
# data and immersed boundary construction parameters to the NetCDF file, as well as
# grid metrics.
#

using Oceananigans.OutputWriters: add_schedule_metadata!, default_output_attributes

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
                grid_index=nothing,
                write_data=true,
                kwargs...)

    # effective_dim_names are the dimensions that will be used to write the field data (excludes reduced and dimensions where location is Nothing)
    effective_dim_names = create_field_dimensions!(ds, fd, dimension_name_generator; time_dependent, with_halos, array_type, dimension_type, grid_index)

    # Add location to attributes
    if :attrib ∈ keys(kwargs)
        attrib = add_location_attribute!(kwargs[:attrib], fd)
    else
        attrib = add_location_attribute!(Dict(), fd)
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

function add_location_attribute!(attrib, fd::AbstractField)
    loc = location(fd) |> convert_for_netcdf
    loc_attrib = Dict("location" => loc)
    return merge(loc_attrib, attrib)
end

#####
##### NetCDFWriter constructor
#####

function NetCDFWriter(model::AbstractModel, outputs;
                      filename,
                      schedule,
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

    # Extract grids from outputs, falling back to model grid for non-field outputs
    output_grids = Dict(name => (try grid(output) catch; grid(model) end) for (name, output) in outputs)
    unique_grids = Tuple(unique(objectid, collect(values(output_grids))))
    output_grid_map = Dict(name => findfirst(gr -> gr === output_grids[name], unique_grids) for name in keys(outputs))

    output_attributes = dictify(output_attributes)
    global_attributes = dictify(global_attributes)
    dimensions = dictify(dimensions)

    # Ensure we can add any kind of metadata to the attributes later by converting to Dict{Any, Any}.
    global_attributes = Dict{Any, Any}(global_attributes)

    dataset, outputs, schedule = initialize_nc_file(model,
                                                    unique_grids,
                                                    output_grid_map,
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

    return NetCDFWriter(unique_grids,
                        output_grid_map,
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
                            grids,
                            output_grid_map,
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

    # Open the NetCDF dataset file
    dataset = NCDataset(filepath, mode, attrib=sort(collect(pairs(global_attributes)), by=first))

    # Only suffix dimension names when there are multiple grids;
    # for a single grid, use nothing so add_grid_suffix is a no-op.
    dim_grid_suffix(idx) = length(grids) == 1 ? nothing : idx

    # Merge default dimension attributes from all unique grids
    all_dim_attributes = Dict()
    for (grid_index, grid) in enumerate(grids)
        merge!(all_dim_attributes, default_dimension_attributes(grid, dimension_name_generator; grid_index=dim_grid_suffix(grid_index)))
    end
    output_attributes = merge(all_dim_attributes, default_output_attributes(model), output_attributes)

    # Define variables for each dimension and attributes if this is a new file.
    if mode == "c"
        # DateTime and TimeDate are both <: AbstractTime
        time_attrib = model.clock.time isa AbstractTime ?
            Dict("long_name" => "Time", "units" => "seconds since 2000-01-01 00:00:00") :
            Dict("long_name" => "Time", "units" => "seconds")

        create_time_dimension!(dataset, attrib=time_attrib, dimension_type=dimension_type)

        # Per-grid: dimensions, reconstruction data, metrics, immersed boundary
        time_independent_vars = Dict()
        time_independent_grid_map = Dict{String, Any}()

        for (grid_index, grid) in enumerate(grids)
            suffix = dim_grid_suffix(grid_index)
            dims = gather_dimensions(outputs, grid, indices, with_halos, dimension_name_generator; grid_index=suffix)
            create_spatial_dimensions!(dataset, dims, output_attributes; deflatelevel=1, dimension_type)

            write_grid_reconstruction_data!(dataset, grid, grid_index; array_type, deflatelevel)
            write_grid_reconstruction_data!(dataset, grid, suffix; array_type, deflatelevel)

            if include_grid_metrics
                metrics = gather_grid_metrics(grid, indices, dimension_name_generator; grid_index=suffix)
                for name in keys(metrics)
                    time_independent_grid_map[name] = suffix
                end
                merge!(time_independent_vars, metrics)
            end

            if grid isa ImmersedBoundaryGrid
                ib_vars = gather_immersed_boundary(grid, indices, dimension_name_generator; grid_index=suffix)
                for name in keys(ib_vars)
                    time_independent_grid_map[name] = suffix
                end
                merge!(time_independent_vars, ib_vars)
            end
        end

        if !isempty(time_independent_vars)
            for (output_name, output) in sort(collect(pairs(time_independent_vars)), by=first)
                grid_index = time_independent_grid_map[output_name]
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
                                        grid_index,
                                        time_dependent = false,
                                        with_halos,
                                        dimension_type)

                save_output!(dataset, output, model, output_name, array_type)
            end
        end

        for (output_name, output) in sort(collect(pairs(outputs)), by=first)
            grid_index = dim_grid_suffix(output_grid_map[output_name])
            attrib = haskey(output_attributes, output_name) ? output_attributes[output_name] : Dict()
            attrib = merge(attrib, Dict("grid_index" => output_grid_map[output_name]))
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
                                    grid_index,
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
                                                                 ow.grids,
                                                                 ow.output_grid_map,
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
                                 time_dependent, with_halos, grid_index=nothing,
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
                                 time_dependent, with_halos, grid_index=nothing,
                                 dimensions, filepath, dimension_type=Float64)

    # If the output is the free surface, we need to handle it differently since it will be writen as a 3D array with a singleton dimension for the z-coordinate
    if output_name == "displacement" && hasfield(typeof(model), :free_surface)
        if output == view(model.free_surface.displacement, output.indices...)
            local default_dimension_name_generator = dimension_name_generator
            dimension_name_generator = (var_name, grid, LX, LY, LZ, dim) -> dimension_name_generator_free_surface(default_dimension_name_generator, var_name, grid, LX, LY, LZ, dim)
        end
    end
    defVar(dataset, output_name, output; array_type, time_dependent, with_halos, dimension_name_generator, deflatelevel, attrib, dimension_type, grid_index, write_data=false)
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
