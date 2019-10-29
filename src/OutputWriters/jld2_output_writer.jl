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

    return nothing
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
    return nothing
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
