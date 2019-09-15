####
#### Output writer utilities
####

ext(fw::AbstractOutputWriter) = throw("Extension for $(typeof(fw)) is not implemented.")

# When saving stuff to disk like a JLD2 file, `saveproperty!` is used, which
# converts Julia objects to language-agnostic objects.
saveproperty!(file, location, p::Number)        = file[location] = p
saveproperty!(file, location, p::AbstractRange) = file[location] = collect(p)
saveproperty!(file, location, p::AbstractArray) = file[location] = Array(p)
saveproperty!(file, location, p::AbstractField)         = file[location] = Array(p.data.parent)
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
                     prefix="", init=noinit, including=[:grid, :eos, :constants, :closure],
                     part=1, max_filesize=Inf, force=false, async=false, verbose=false)

Construct an `OutputWriter` that writes `label, func` pairs in the dictionary `outputs` to
a single JLD2 file, where `label` is a symbol that labels the output and `func` is a
function of the form `func(model)` that returns the data to be saved.

Keyword arguments
=================

    - `frequency::Int`   : Save output every `n` model iterations.
    - `interval::Int`    : Save output every `t` units of model clock time.
    - `dir::String`      : Directory to save output to. Default: "." (current working
                           directory).
    - `prefix::String`   : Descriptive filename prefixed to all output files. Default: "".
    - `init::Function`   : A function of the form `init(file, model)` that runs when a JLD2
                           output file is initialized. Default: `noinit(args...) = nothing`.
    - `including::Array` : List of model properties to save with every file. By default, the
                           grid, equation of state, planetary constants, and the turbulence
                           closure parameters are saved.
    - `part::Int`        : The starting part number used if `max_filesize` is finite.
                           Default: 1.
    - `max_filesize::Int`: The writer will stop writing to the output file once the file size
                           exceeds `max_filesize`, and write to a new one with a consistent
                           naming scheme ending in `part1`, `part2`, etc. Defaults to `Inf`.
    - `force::Bool`      : Remove existing files if their filenames conflict. Default: `false`.
    - `async::Bool`      : Write output asynchronously. Default: `false`.
    - `verbose::Bool`    : Log what the output writer is doing with statistics on compute/write
                           times and file sizes. Default: `false`.
    - `jld2_kw::Dict`    : Dict of kwargs to be passed to `jldopen` when data is written.
"""
function JLD2OutputWriter(model, outputs; interval=nothing, frequency=nothing, dir=".", prefix="",
                          init=noinit, including=[:grid, :eos, :constants, :closure],
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

"A type for writing NetCDF output."
Base.@kwdef mutable struct NetCDFOutputWriter <: AbstractOutputWriter
             dir  :: AbstractString = "."
          prefix  :: AbstractString = ""
        frequency :: Int    = 1
          padding :: Int    = 9
    naming_scheme :: Symbol = :iteration
      compression :: Int    = 3
            async :: Bool   = false
end

ext(fw::NetCDFOutputWriter) = ".nc"

function filename(fw, name, iteration)
    if fw.naming_scheme == :iteration
        fw.prefix * name * lpad(iteration, fw.padding, "0") * ext(fw)
    elseif fw.naming_scheme == :file_number
        file_num = Int(iteration / fw.frequency)
        fw.prefix * name * lpad(file_num, fw.padding, "0") * ext(fw)
    else
        throw(ArgumentError("Invalid naming scheme: $(fw.naming_scheme)"))
    end
end

function write_output(model::Model, fw::NetCDFOutputWriter)
    fields = Dict(
        "xC" => collect(model.grid.xC),
        "yC" => collect(model.grid.yC),
        "zC" => collect(model.grid.zC),
        "xF" => collect(model.grid.xF),
        "yF" => collect(model.grid.yF),
        "zF" => collect(model.grid.zF),
        "u" => Array(parentdata(model.velocities.u)),
        "v" => Array(parentdata(model.velocities.v)),
        "w" => Array(parentdata(model.velocities.w)),
        "T" => Array(parentdata(model.tracers.T)),
        "S" => Array(parentdata(model.tracers.S))
    )

    if fw.async
        # Execute asynchronously on worker 2.
        i = model.clock.iteration
        @async remotecall(write_output_netcdf, 2, fw, fields, i)
    else
        write_output_netcdf(fw, fields, model.clock.iteration)
    end

    return
end

function write_output_netcdf(fw::NetCDFOutputWriter, fields, iteration)
    xC, yC, zC = fields["xC"], fields["yC"], fields["zC"]
    xF, yF, zF = fields["xF"], fields["yF"], fields["zF"]

    u, v, w = fields["u"], fields["v"], fields["w"]
    T, S    = fields["T"], fields["S"]

    xC_attr = Dict("longname" => "Locations of the cell centers in the x-direction.", "units" => "m")
    yC_attr = Dict("longname" => "Locations of the cell centers in the y-direction.", "units" => "m")
    zC_attr = Dict("longname" => "Locations of the cell centers in the z-direction.", "units" => "m")

    xF_attr = Dict("longname" => "Locations of the cell faces in the x-direction.", "units" => "m")
    yF_attr = Dict("longname" => "Locations of the cell faces in the y-direction.", "units" => "m")
    zF_attr = Dict("longname" => "Locations of the cell faces in the z-direction.", "units" => "m")

    u_attr = Dict("longname" => "Velocity in the x-direction", "units" => "m/s")
    v_attr = Dict("longname" => "Velocity in the y-direction", "units" => "m/s")
    w_attr = Dict("longname" => "Velocity in the z-direction", "units" => "m/s")
    T_attr = Dict("longname" => "Temperature", "units" => "K")
    S_attr = Dict("longname" => "Salinity", "units" => "g/kg")

    filepath = joinpath(fw.dir, filename(fw, "", iteration))

    if fw.async
        println("[Worker $(Distributed.myid()): NetCDFOutputWriter] Writing fields to disk: $filepath")
    end

    isfile(filepath) && rm(filepath)

    nccreate(filepath, "u", "xF", xC, xC_attr,
                            "yC", yC, yC_attr,
                            "zC", zC, zC_attr,
                            atts=u_attr, compress=fw.compression)

    nccreate(filepath, "v", "xC", xC, xC_attr,
                            "yF", yC, yC_attr,
                            "zC", zC, zC_attr,
                            atts=v_attr, compress=fw.compression)

    nccreate(filepath, "w", "xC", xC, xC_attr,
                            "yC", yC, yC_attr,
                            "zF", zC, zC_attr,
                            atts=w_attr, compress=fw.compression)

    nccreate(filepath, "T", "xC", xC, xC_attr,
                            "yC", yC, yC_attr,
                            "zC", zC, zC_attr,
                            atts=T_attr, compress=fw.compression)

    nccreate(filepath, "S", "xC", xC, xC_attr,
                            "yC", yC, yC_attr,
                            "zC", zC, zC_attr,
                            atts=S_attr, compress=fw.compression)

    ncwrite(u, filepath, "u")
    ncwrite(v, filepath, "v")
    ncwrite(w, filepath, "w")
    ncwrite(T, filepath, "T")
    ncwrite(S, filepath, "S")

    ncclose(filepath)

    return
end

function read_output(fw::NetCDFOutputWriter, field_name, iter)
    filepath = joinpath(fw.dir, filename(fw, "", iter))
    field_data = ncread(filepath, field_name)
    ncclose(filepath)
    return field_data
end

####
#### Checkpointer
####

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

function Checkpointer(model; frequency=nothing, interval=nothing, dir=".", prefix="checkpoint", force=false,
                      verbose=false, properties = [:architecture, :boundary_conditions, :grid, :clock, :eos, :constants,
                                                   :closure, :velocities, :tracers, :timestepper])

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
