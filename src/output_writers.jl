####
#### Output writer utilities
####

ext(fw::AbstractOutputWriter) = throw("Extension for $(typeof(fw)) is not implemented.")

# When saving stuff to disk like a JLD2 file, `saveproperty!` is used, which
# converts Julia objects to language-agnostic objects.
saveproperty!(file, location, p::Number)        = file[location] = p
saveproperty!(file, location, p::AbstractRange) = file[location] = collect(p)
saveproperty!(file, location, p::AbstractArray) = file[location] = Array(p)
saveproperty!(file, location, p::AbstractField) = file[location] = Array(p.data.parent)
saveproperty!(file, location, p::Function) = @warn "Cannot save Function property into $location"
saveproperty!(file, location, p) = [saveproperty!(file, location * "/$subp", getproperty(p, subp))
                                        for subp in propertynames(p)]

# Special saveproperty! so boundary conditions are easily readable outside julia.
function saveproperty!(file, location, cbcs::CoordinateBoundaryConditions)
    for endpoint in propertynames(cbcs)
        endpoint_bc = getproperty(cbcs, endpoint)
        if isa(endpoint_bc.condition, Function)
            @warn "$field.$coord.$endpoint boundary is of type Function and cannot be saved to disk!"
            file["boundary_conditions/$field/$coord/$endpoint/type"] = string(bctype(endpoint_bc))
            file["boundary_conditions/$field/$coord/$endpoint/condition"] = missing
        else
            file["boundary_conditions/$field/$coord/$endpoint/type"] = string(bctype(endpoint_bc))
            file["boundary_conditions/$field/$coord/$endpoint/condition"] = endpoint_bc.condition
        end
    end
end

# Special savepropety! for AB2 time stepper struct used by the checkpointer so
# it only saves the fields and not the tendency BCs or χ value (as they can be
# constructed by the `Model` constructor).
function saveproperty!(file, location, ts::AdamsBashforthTimestepper)
    saveproperty!(file, location * "/Gⁿ", ts.Gⁿ)
    saveproperty!(file, location * "/G⁻", ts.G⁻)
end

saveproperties!(file, structure, ps) = [saveproperty!(file, "$p", getproperty(structure, p)) for p in ps]

# When checkpointing, `serializeproperty!` is used, which serializes objects
# unless they need to be converted (basically CuArrays only).
serializeproperty!(file, location, p) = file[location] = p
serializeproperty!(file, location, p::Union{AbstractArray, AbstractField}) = saveproperty!(file, location, p)
serializeproperty!(file, location, p::Function) = @warn "Cannot serialize Function property into $location"

serializeproperties!(file, structure, ps) = [serializeproperty!(file, "$p", getproperty(structure, p)) for p in ps]

# Don't check arrays because we don't need that noise.
has_reference(T, ::AbstractArray{<:Number}) = false

# These two conditions are true, but should not necessary.
has_reference(::Type{Function}, ::AbstractField) = false
has_reference(::Type{T}, ::NTuple{N, <:T}) where {N, T} = true

"""
    has_reference(has_type, obj)

Check (or attempt to check) if `obj` contains, somewhere among its
subfields and subfields of fields, a reference to an object of type
`has_type`. This function doesn't always work.
"""
function has_reference(has_type, obj)
    if typeof(obj) <: has_type
        return true
    elseif applicable(iterate, obj) && length(obj) > 1
        return any([has_reference(has_type, elem) for elem in obj])
    elseif applicable(propertynames, obj) && length(propertynames(obj)) > 0
        return any([has_reference(has_type, getproperty(obj, p)) for p in propertynames(obj)])
    else
        return typeof(obj) <: has_type
    end
end

####
####  JLD2 output writer
####

"""
    JLD2OutputWriter{F, I, O, IF, IN, KW} <: AbstractOutputWriter

An output writer for writing to JLD2 files.
"""
mutable struct JLD2OutputWriter{F, I, O, IF, IN, KW} <: AbstractOutputWriter
        filepath :: String
         outputs :: O
        interval :: I
       frequency :: F
            init :: IF
       including :: IN
        previous :: Float64
            part :: Int
    max_filesize :: Float64
           async :: Bool
           force :: Bool
         verbose :: Bool
         jld2_kw :: KW
end

noinit(args...) = nothing

"""
    JLD2OutputWriter(model, outputs; interval=nothing, frequency=nothing, dir=".",
                     prefix="", init=noinit, including=[:grid, :coriolis, :buoyancy, :closure],
                     part=1, max_filesize=Inf, force=false, async=false, verbose=false)

Construct a `JLD2OutputWriter` that writes `label, func` pairs in `outputs` (which can be a `Dict` or `NamedTuple`)
to a JLD2 file, where `label` is a symbol that labels the output and `func` is a function of the form `func(model)`
that returns the data to be saved.

Keyword arguments
=================
- `frequency::Int`   : Save output every `n` model iterations.
- `interval::Int`    : Save output every `t` units of model clock time.
- `dir::String`      : Directory to save output to. Default: "." (current working directory).
- `prefix::String`   : Descriptive filename prefixed to all output files. Default: "".
- `init::Function`   : A function of the form `init(file, model)` that runs when a JLD2 output file is initialized.
                       Default: `noinit(args...) = nothing`.
- `including::Array` : List of model properties to save with every file. By default, the grid, equation of state,
                       Coriolis parameters, buoyancy parameters, and turbulence closure parameters are saved.
- `part::Int`        : The starting part number used if `max_filesize` is finite. Default: 1.
- `max_filesize::Int`: The writer will stop writing to the output file once the file size exceeds `max_filesize`, and
                       write to a new one with a consistent naming scheme ending in `part1`, `part2`, etc. Defaults to
                       `Inf`.
- `force::Bool`      : Remove existing files if their filenames conflict. Default: `false`.
- `async::Bool`      : Write output asynchronously. Default: `false`.
- `verbose::Bool`    : Log what the output writer is doing with statistics on compute/write times and file sizes.
                       Default: `false`.
- `jld2_kw::Dict`    : Dict of kwargs to be passed to `jldopen` when data is written.
"""
function JLD2OutputWriter(model, outputs; interval=nothing, frequency=nothing, dir=".", prefix="",
                          init=noinit, including=[:grid, :coriolis, :buoyancy, :closure],
                          part=1, max_filesize=Inf, force=false, async=false, verbose=false,
                          jld2_kw=Dict{Symbol, Any}())

    validate_interval(interval, frequency)

    mkpath(dir)
    filepath = joinpath(dir, prefix * ".jld2")
    force && isfile(filepath) && rm(filepath, force=true)

    jldopen(filepath, "a+"; jld2_kw...) do file
        init(file, model)
        saveproperties!(file, model, including)
    end

    return JLD2OutputWriter(filepath, outputs, interval, frequency, init, including,
                            0.0, part, max_filesize, async, force, verbose, jld2_kw)
end

function write_output(model, fw::JLD2OutputWriter)
    verbose = fw.verbose
    verbose && @info @sprintf("Calculating JLD2 output %s...", keys(fw.outputs))
    tc = Base.@elapsed data = Dict((name, f(model)) for (name, f)
                                    in zip(keys(fw.outputs), values(fw.outputs)))
    verbose && @info "Calculation time: $(prettytime(tc))"

    iter = model.clock.iteration
    time = model.clock.time

    filesize(fw.filepath) >= fw.max_filesize && start_next_file(model, fw)

    path = fw.filepath
    verbose && @info "Writing JLD2 output $(keys(fw.outputs)) to $path..."
    t0, sz = time_ns(), filesize(path)

    if fw.async
        @async remotecall(jld2output!, 2, path, iter, time, data, fw.jld2_kw)
    else
        jld2output!(path, iter, time, data, fw.jld2_kw)
    end

    t1, newsz = time_ns(), filesize(path)

    verbose && @info @sprintf("Writing done: time=%s, size=%s, Δsize=%s",
                              prettytime((t1-t0)/1e9), pretty_filesize(newsz), pretty_filesize(newsz-sz))

    return
end

"""
    jld2output!(path, iter, time, data, kwargs)

Write the (name, value) pairs in `data`, including the simulation
`time`, to the JLD2 file at `path` in the `timeseries` group,
stamping them with `iter` and using `kwargs` when opening
the JLD2 file.
"""
function jld2output!(path, iter, time, data, kwargs)
    jldopen(path, "r+"; kwargs...) do file
        file["timeseries/t/$iter"] = time
        for (name, datum) in data
            file["timeseries/$name/$iter"] = datum
        end
    end
    return
end

function start_next_file(model::Model, fw::JLD2OutputWriter)
    verbose = fw.verbose
    sz = filesize(fw.filepath)
    verbose && @info begin
        "Filesize $(pretty_filesize(sz)) has exceeded maximum file size $(pretty_filesize(fw.max_filesize))."
    end

    if fw.part == 1
        part1_path = replace(fw.filepath, r".jld2$" => "_part1.jld2")
        verbose && @info "Renaming first part: $(fw.filepath) -> $part1_path"
        mv(fw.filepath, part1_path, force=fw.force)
        fw.filepath = part1_path
    end

    fw.part += 1
    fw.filepath = replace(fw.filepath, r"part\d+.jld2$" => "part" * string(fw.part) * ".jld2")
    fw.force && isfile(fw.filepath) && rm(fw.filepath, force=true)
    verbose && @info "Now writing to: $(fw.filepath)"

    jldopen(fw.filepath, "a+"; fw.jld2_kw...) do file
        fw.init(file, model)
        saveproperties!(file, model, fw.including)
    end
end

"""
    FieldOutput([return_type=Array], field)

Returns a `FieldOutput` type intended for use with the `JLD2OutputWriter`.
Calling `FieldOutput(model)` returns `return_type(field.data.parent)`.
"""
struct FieldOutput{O, F}
    return_type :: O
          field :: F
end

FieldOutput(field) = FieldOutput(Array, field) # default
(fo::FieldOutput)(model) = fo.return_type(fo.field.data.parent)

"""
    FieldOutputs(fields)

Returns a dictionary of `FieldOutput` objects with key, value
pairs corresponding to each name and value in the `NamedTuple` `fields`.
Intended for use with `JLD2OutputWriter`.

Examples
========

```julia
julia> output_writer = JLD2OutputWriter(model, FieldOutputs(model.velocities), frequency=1);

julia> keys(output_writer.outputs)
Base.KeySet for a Dict{Symbol,FieldOutput{UnionAll,F} where F} with 3 entries. Keys:
  :w
  :v
  :u
```
"""
function FieldOutputs(fields)
    names = propertynames(fields)
    nfields = length(fields)
    return Dict((names[i], FieldOutput(fields[i])) for i in 1:nfields)
end

####
#### NetCDF output writer
####

"""
    netcdf_spatial_dimensions(::field)

Returns the dimensions associated with a field.

Examples
========

julia> netcdf_spatial_dimensions(model.velocities.u)
("xF", "yC", "zC")

julia> netcdf_spatial_dimensions(model.tracers.T)
("xC", "yC", "zC")
"""
function netcdf_spatial_dimensions(field::Field{LX, LY, LZ}) where {LX, LY, LZ}
    xdim(LX), ydim(LY), zdim(LZ)
end

xdim(::Type{Face}) = "xF"
xdim(::Type{Cell}) = "xC"
ydim(::Type{Face}) = "yF"
ydim(::Type{Cell}) = "yC"
zdim(::Type{Face}) = "zF"
zdim(::Type{Cell}) = "zC"

# This function converts and integer to a range, and nothing to a Colon
get_slice(n::Integer) = n:n
get_slice(n::UnitRange) = n
get_slice(n::Nothing) = Colon()

"""
    write_grid(model; filename="grid.nc", mode="c", 
               compression=0, attributes=Dict(), slice_kw...)

Writes a grid.nc file that contains all the dimensions of the domain.

Keyword arguments
=================

    - `filename::String`  : File name to be saved under
    - `mode::String`      : Netcdf file is opened in either clobber ("c") or append ("a") mode (Default: "c" )
    - `compression::Int`  : Defines the compression level of data (0-9, default 0)
    - `attributes::Dict`  : Attributes (default: Dict())
"""
function write_grid(model; filename="./grid.nc", mode="c",
                    compression=0, attributes=Dict(), slice_kw...)
    dimensions = Dict(
        "xC" => collect(model.grid.xC),
        "yC" => collect(model.grid.yC),
        "zC" => collect(model.grid.zC),
        "xF" => collect(model.grid.xF)[1:end-1],
        "yF" => collect(model.grid.yF)[1:end-1],
        "zF" => collect(model.grid.zF)[1:end-1]
    )
    dim_attrib = Dict(
        "xC" => ["longname" => "Locations of the cell centers in the x-direction.", "units" => "m"],
        "yC" => ["longname" => "Locations of the cell centers in the y-direction.", "units" => "m"],
        "zC" => ["longname" => "Locations of the cell centers in the z-direction.", "units" => "m"],
        "xF" => ["longname" => "Locations of the cell faces in the x-direction.",   "units" => "m"],
        "yF" => ["longname" => "Locations of the cell faces in the y-direction.",   "units" => "m"],
        "zF" => ["longname" => "Locations of the cell faces in the z-direction.",   "units" => "m"]
    )

    # Applies slices to the dimensions d
    for (d, slice) in slice_kw
        if String(d) in keys(dimensions)
            dimensions[String(d)] = dimensions[String(d)][get_slice(slice)]
        end
    end

    # Writes the sliced dimensions to the specified netcdf file
    Dataset(filename, mode, attrib=attributes) do ds
        for (dimname, dimarray) in dimensions
            defVar(ds, dimname, dimarray, (dimname,),
                   compression=compression, attrib=dim_attrib[dimname])
        end
    end
    return
end


"""
    NetCDFOutputWriter <: AbstractOutputWriter

An output writer for writing to NetCDF files.
"""
mutable struct NetCDFOutputWriter <: AbstractOutputWriter
             filename :: String
              dataset :: Any
              outputs :: Dict
             interval :: Union{Nothing, AbstractFloat}
            frequency :: Union{Nothing, Int}
              clobber :: Bool
               slices :: Dict
   len_time_dimension :: Int
             previous :: Float64
end

"""
    NetCDFOutputWriter(model, outputs; interval=nothing, frequency=nothing, filename=".",
                                   clobber=true, global_attributes=Dict(), output_attributes=nothing, slice_kw...)

Construct a `NetCDFOutputWriter` that writes `label, field` pairs in `outputs` (which can be a
`Dict` or `NamedTuple`) to a NC file, where `label` is a symbol that labels the output and
`field` is a field from the model (e.g. `model.velocities.u`).

Keyword arguments
=================

    - `filename::String`         : Directory to save output to. Default: "." (current working directory).
    - `frequency::Int`           : Save output every `n` model iterations.
    - `interval::Int`            : Save output every `t` units of model clock time.
    - `clobber::Bool`            : Remove existing files if their filenames conflict. Default: `false`.
    - `compression::Int`         : Determines the compression level of data (0-9, default 0)
    - `global_attributes::Dict`  : Dict of model properties to save with every file (deafult: Dict())
    - `output_attributes::Dict`  : Dict of attributes to be saved with each field variable (reasonable
                                   defaults are provided for velocities, temperature, and salinity)
    - `slice_kw`                 : `dimname = Union{OrdinalRange, Integer}` will slice the dimension `dimname`.
                                   All other keywords are ignored.
                                   e.g. `xC = 3:10` will only produce output along the dimension `xC` between
                                   indices 3 and 10 for all fields with `xC` as one of their dimensions.
                                   `xC = 1` is treated like `xC = 1:1`.
                                   Multiple dimensions can be sliced in one call. Not providing slices writes
                                   output over the entire domain.
"""

function NetCDFOutputWriter(model, outputs; interval=nothing, frequency=nothing, filename=".",
                        clobber=true, global_attributes=Dict(), output_attributes=nothing, compression=0, slice_kw...)

    validate_interval(interval, frequency)

    mode = clobber ? "c" : "a"

    # Initiates the output file with dimensions
    write_grid(model; filename=filename, mode=mode,
               compression=compression, attrib=global_attributes, slice_kw...)

    # Opens the same output file for writing fields from the user-supplied variable outputs
    dataset = Dataset(filename, "a")

    # Creates an unliimited dimension "Time"
    defDim(dataset, "Time", Inf); sync(dataset)
    defVar(dataset, "Time", Float64, ("Time",)); sync(dataset)
    len_time_dimension = 0 # Number of time-steps so far

    if isa(output_attributes, Nothing)
        output_attributes = Dict("u" => Dict("longname" => "Velocity in the x-direction", "units" => "m/s"),
                            "v" => Dict("longname" => "Velocity in the y-direction", "units" => "m/s"),
                            "w" => Dict("longname" => "Velocity in the z-direction", "units" => "m/s"),
                            "T" => Dict("longname" => "Temperature", "units" => "K"),
                            "S" => Dict("longname" => "Salinity", "units" => "g/kg"))
    end

    # Initiates empty Float32 arrays for fields from the user-supplied variable outputs
    for (fieldname, field) in outputs
        dtype = eltype(parentdata(field))
        defVar(dataset, fieldname, dtype, (netcdf_spatial_dimensions(field)...,"Time"),
               compression=compression, attrib=output_attributes[fieldname])
    end
    sync(dataset)

    # Stores slices for the dimensions of each output field
    slices = Dict{String, Vector{Union{OrdinalRange,Colon}}}()
    for (fieldname, field) in outputs
        slices[fieldname] = slice(field; slice_kw...)
    end

    return NetCDFOutputWriter(filename, dataset, outputs, interval,
                          frequency, clobber, slices, len_time_dimension, 0.0)
end


# Closes the outputwriter
Base.close(fw::NetCDFOutputWriter) = close(fw.dataset)

function read_output(fw::NetCDFOutputWriter, fieldname)
    ds = Dataset(fw.filename,"r")
    field = ds[fieldname][:,:,:,end]
    close(ds)
    return field
end

"""
    slice(field; slice_kw...)

For internal use only. Returns a slice for a field based on its dimensions and the supplied slices in `slice_kw`.
"""
function slice(field; slice_kw...)
    slice_array = Vector{Union{AbstractRange,Colon}}()
    for dim in netcdf_spatial_dimensions(field)
        slice = get_slice(haskey(slice_kw, Symbol(dim)) ? slice_kw[Symbol(dim)] : nothing)
        push!(slice_array, slice)
    end
    return slice_array
end

"""
    write_output(model, OutputWriter)

For internal user only. Writes output to the netcdf file at specified intervals. Increments the `Time` dimension every time an output is written to the file.
"""
function write_output(model, fw::NetCDFOutputWriter)
    fw.len_time_dimension += 1
    fw.dataset["Time"][fw.len_time_dimension] = model.clock.time
    for (fieldname, field) in fw.outputs
        fw.dataset[fieldname][:,:,:,fw.len_time_dimension] = view(parentdata(field), fw.slices[fieldname]...)
    end
    sync(fw.dataset)
    return
end

####
#### Checkpointer
####

"""
    Checkpointer{I, T, P, A} <: AbstractOutputWriter

An output writer for checkpointing models to a JLD2 file from which models can be restored.
"""
mutable struct Checkpointer{I, T, P, A} <: AbstractOutputWriter
         frequency :: I
          interval :: T
          previous :: Float64
               dir :: String
            prefix :: String
        properties :: P
    has_array_refs :: A
             force :: Bool
           verbose :: Bool
end

"""
    Checkpointer(model; frequency=nothing, interval=nothing, dir=".", prefix="checkpoint",
                        force=false, verbose=false,
                        properties = [:architecture, :boundary_conditions, :grid, :clock,
                                      :eos, :constants, :closure, :velocities, :tracers,
                                      :timestepper])

Construct a `Checkpointer` that checkpoints the model to a JLD2 file every so often as
specified by `frequency` or `interval`. The `model.clock.iteration` is included in the
filename to distinguish between multiple checkpoint files.

Note that extra model `properties` can be safely specified, but removing crucial properties
such as `:velocities` will make restoring from the checkpoint impossible.

The checkpoint file is generated by serializing model properties to JLD2. However,
functions cannot be serialized to disk (at least not with JLD2). So if a model property
contains a reference somewhere in its hierarchy it will not be included in the checkpoint
file (and you will have to manually restore them).

Keyword arguments
=================
- `frequency::Int`   : Save output every `n` model iterations.
- `interval::Int`    : Save output every `t` units of model clock time.
- `dir::String`      : Directory to save output to. Default: "." (current working directory).
- `prefix::String`   : Descriptive filename prefixed to all output files. Default: "checkpoint".
- `force::Bool`      : Remove existing files if their filenames conflict. Default: `false`.
- `verbose::Bool`    : Log what the output writer is doing with statistics on compute/write times and file sizes.
                       Default: `false`.
- `properties::Array`: List of model properties to checkpoint.
"""
function Checkpointer(model; frequency=nothing, interval=nothing, dir=".", prefix="checkpoint", force=false,
                      verbose=false, properties = [:architecture, :boundary_conditions, :grid, :clock, :coriolis,
                                                   :buoyancy, :closure, :velocities, :tracers, :timestepper])

    validate_interval(frequency, interval)

    # Grid needs to be checkpointed for restoring to work.
    :grid ∉ properties && push!(properties, :grid)

    has_array_refs = Symbol[]

    for p in properties
        isa(p, Symbol) || @error "Property $p to be checkpointed must be a Symbol."
        p ∉ propertynames(model) && @error "Cannot checkpoint $p, it is not a model property!"

        if has_reference(Function, getproperty(model, p))
            @warn "model.$p contains a function somewhere in its hierarchy and will not be checkpointed."
            filter!(e -> e != p, properties)
        end

        has_reference(AbstractField, getproperty(model, p)) && push!(has_array_refs, p)
    end

    mkpath(dir)
    return Checkpointer(frequency, interval, 0.0, dir, prefix, properties, has_array_refs, force, verbose)
end

function write_output(model, c::Checkpointer)
    filepath = joinpath(c.dir, c.prefix * string(model.clock.iteration) * ".jld2")
    c.verbose && @info "Checkpointing to file $filepath..."

    t0 = time_ns()
    jldopen(filepath, "w") do file
        file["checkpointed_properties"] = c.properties
        file["has_array_refs"] = c.has_array_refs

        serializeproperties!(file, model, filter(e -> !(e ∈ c.has_array_refs), c.properties))
        saveproperties!(file, model, filter(e -> e ∈ c.has_array_refs, c.properties))
    end

    t1, sz = time_ns(), filesize(filepath)
    c.verbose && @info "Checkpointing done: time=$(prettytime((t1-t0)/1e9)), size=$(pretty_filesize(sz))"
end

defaultname(::Checkpointer, nelems) = :checkpointer
_arr(::CPU, a) = a
_arr(::GPU, a) = CuArray(a)

function restore_fields!(model, file, arch, fieldset; location="$fieldset")
    if fieldset == :timestepper
        restore_fields!(model.timestepper, file, arch, :Gⁿ; location="timestepper/Gⁿ")
        restore_fields!(model.timestepper, file, arch, :G⁻; location="timestepper/G⁻")
    else
        for p in propertynames(getproperty(model, fieldset))
            getproperty(getproperty(model, fieldset), p).data.parent .= _arr(arch, file[location * "/$p"])
        end
    end
end

"""
    restore_from_checkpoint(filepath; kwargs=Dict())

Restore a model from the checkpoint file stored at `filepath`. `kwargs` can be passed
to the `Model` constructor, which can be especially useful if you need to manually
restore forcing functions or boundary conditions that rely on functions.
"""
function restore_from_checkpoint(filepath; kwargs=Dict())
    file = jldopen(filepath, "r")

    cps = file["checkpointed_properties"]
    has_array_refs = file["has_array_refs"]

    # Restore model properties that were just serialized.
    # We skip fields that contain structs and restore them after model creation.
    for p in cps
        if p ∉ has_array_refs
            kwargs[p] = file["$p"]
        end
    end

    model = Model(; kwargs...)

    # Now restore fields.
    for p in cps
        if p in has_array_refs
            restore_fields!(model, file, model.architecture, p)
        end
    end

    close(file)

    return model
end

