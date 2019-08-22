using Distributed
using NetCDF, JLD2

ext(fw::OutputWriter) = throw("Extension for $(typeof(fw)) is not implemented.")

function time_to_run(clock::Clock, diag::OutputWriter)
    if :interval in propertynames(diag) && diag.interval != nothing
        if clock.time >= diag.previous + diag.interval
            diag.previous = clock.time - rem(clock.time, diag.interval)
            return true
        else
            return false
        end
    elseif :frequency in propertynames(diag) && diag.frequency != nothing
        return clock.iteration % diag.frequency == 0
    else
        error("$(typeof(diag)) must have a frequency or interval specified!")
    end
end

function validate_interval(frequency, interval)
    if isnothing(frequency) && isnothing(interval)
        error("Must choose either a frequency (number of iterations) or a time interval!")
    elseif isnothing(interval)
        if isinteger(frequency)
            return Int(frequency), interval
        else
            error("Frequency $frequency must be an integer!")
        end
    elseif isnothing(frequency)
        if isa(interval, Number)
            return frequency, Float64(interval)
        else
            error("Interval must be convertable to a float!")
        end
    else
        error("Cannot choose both frequency and interval!")
    end
end

####
#### Binary output writer
####

"A type for writing Binary output."
Base.@kwdef mutable struct BinaryOutputWriter <: OutputWriter
          dir :: AbstractString = "."
       prefix :: AbstractString = ""
    frequency :: Int = 1
      padding :: Int = 9
end

function write_output(model::Model, fw::BinaryOutputWriter)
    for (field, field_name) in zip(fw.fields, fw.field_names)
        filepath = joinpath(fw.dir, filename(fw, field_name, model.clock.iteration))

        println("[BinaryOutputWriter] Writing $field_name to disk: $filepath")
        if model.metadata == :CPU
            write(filepath, field.data)
        elseif model.metadata == :GPU
            write(filepath, Array(field.data))
        end
    end
end

function read_output(model::Model, fw::BinaryOutputWriter, field_name, time)
    filepath = joinpath(fw.dir, filename(fw, field_name, time_step))
    arr = zeros(model.metadata.float_type, model.grid.Nx, model.grid.Ny, model.grid.Nz)

    open(filepath, "r") do fio
        read!(fio, arr)
    end

    return arr
end

####
#### NetCDF output writer
####

"A type for writing NetCDF output."
Base.@kwdef mutable struct NetCDFOutputWriter <: OutputWriter
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
        "u" => Array(ardata(model.velocities.u)),
        "v" => Array(ardata(model.velocities.v)),
        "w" => Array(ardata(model.velocities.w)),
        "T" => Array(ardata(model.tracers.T)),
        "S" => Array(ardata(model.tracers.S))
    )

    if fw.async
        # Execute asynchronously on worker 2.
        i = model.clock.iteration
        @async remotecall(write_output_netcdf, 2, fw, fields, i)
    else
        write_output_netcdf(fw, fields, model.clock.iteration)
    end

    return nothing
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

    return nothing
end

function read_output(fw::NetCDFOutputWriter, field_name, iter)
    filepath = joinpath(fw.dir, filename(fw, "", iter))
    field_data = ncread(filepath, field_name)
    ncclose(filepath)
    return field_data
end

####
####  JLD2 output writer
####

mutable struct JLD2OutputWriter{F, I, O, IF, IN} <: OutputWriter
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
end

noinit(args...) = nothing

"""
    JLD2OutputWriter(model, outputs; interval=nothing, frequency=nothing, dir=".",
                     prefix="", init=noinit, including=[:grid, :eos, :constants, :closure],
                     part=1, max_filesize=Inf, force=false, async=false, verbose=false)

Construct an `OutputWriter` that writes `label, func` pairs in the dictionary `outputs` to
a single JLD2 file, where `label` is a symbol that labels the output and `func` is a
function of the form `func(model)` that returns the data to be saved.

# Keyword arguments
- `frequency::Int`:    Save output every `n` model iterations.
- `interval::Int`:     Save output every `t` units of model clock time.
- `dir::String`:       Directory to save output to. Default: "." (current working
                       directory).
- `prefix::String`:    Descriptive filename prefixed to all output files. Default: "".
- `init::Function`:    A function of the form `init(file, model)` that runs when a JLD2
                       output file is initialized. Default: `noinit(args...) = nothing`.
- `including::Array`:  List of model properties to save with every file. By default, the
                       grid, equation of state, planetary constants, and the turbulence
                       closure parameters are saved.
- `part::Int`:         The starting part number used if `max_filesize` is finite.
                       Default: 1.
- `max_filesize::Int`: The writer will stop writing to the output file once the file size
                       exceeds `max_filesize`, and write to a new one with a consistent
                       naming scheme ending in `part1`, `part2`, etc. Defaults to `Inf`.
- `force::Bool`:       Remove existing files if their filenames conflict. Default: `false`.
- `async::Bool`:       Write output asynchronously. Default: `false`.
- `verbose::Bool`:     Log what the output writer is doing with statistics on compute/write
                       times and file sizes. Default: `false`.
"""
function JLD2OutputWriter(model, outputs; interval=nothing, frequency=nothing, dir=".", prefix="",
                          init=noinit, including=[:grid, :eos, :constants, :closure],
                          part=1, max_filesize=Inf, force=false, async=false, verbose=false)

    interval === nothing && frequency === nothing &&
        error("Either interval or frequency must be passed to the JLD2OutputWriter!")

    mkpath(dir)
    filepath = joinpath(dir, prefix * ".jld2")
    force && isfile(filepath) && rm(filepath, force=true)

    jldopen(filepath, "a+") do file
        init(file, model)
        savesubstructs!(file, model, including)
    end

    return JLD2OutputWriter(filepath, outputs, interval, frequency, init, including,
                            0.0, part, max_filesize, async, force, verbose)
end

function savesubstruct!(file, model, name, flds=propertynames(getproperty(model, name)))
    for fld in flds
        file["$name/$fld"] = getproperty(getproperty(model, name), fld)
    end
    return nothing
end

savesubstructs!(file, model, names) = [savesubstruct!(file, model, name) for name in names]

function write_output(model, fw::JLD2OutputWriter)
    verbose = fw.verbose
    verbose && @info @sprintf("Calculating JLD2 output %s...", keys(fw.outputs))
    tc = Base.@elapsed data = Dict((name, f(model)) for (name, f) in fw.outputs)
    verbose && @info "Calculation time: $(prettytime(tc))"

    iter = model.clock.iteration
    time = model.clock.time

    filesize(fw.filepath) >= fw.max_filesize && start_next_file(model, fw)

    path = fw.filepath
    verbose && @info "Writing JLD2 output $(keys(fw.outputs))..."
    t0, sz = time_ns(), filesize(path)

    if fw.async
        @async remotecall(jld2output!, 2, path, iter, time, data)
    else
        jld2output!(path, iter, time, data)
    end

    t1, newsz = time_ns(), filesize(path)
    verbose && @info "Writing done: time=$(prettytime((t1-t0)/1e9)), size=$(pretty_filesize(newsz)), Δsize=$(pretty_filesize(newsz-sz))"

    return nothing
end

function start_next_file(model::Model, fw::JLD2OutputWriter)
    verbose = fw.verbose
    sz = filesize(fw.filepath)
    verbose && @info "Filesize $(pretty_filesize(sz)) has exceeded maximum file size $(pretty_filesize(fw.max_filesize))."

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

    jldopen(fw.filepath, "a+") do file
        fw.init(file, model)
        savesubstructs!(file, model, fw.including)
    end
end

function jld2output!(path, iter, time, data)
    jldopen(path, "r+") do file
        file["timeseries/t/$iter"] = time
        for (name, datum) in data
            file["timeseries/$name/$iter"] = datum
        end
    end
    return nothing
end

function time_to_write(clock, out::JLD2OutputWriter{<:Nothing})
    if clock.time > out.previous + out.interval
        out.previous = clock.time - rem(clock.time, out.interval)
        return true
    else
        return false
    end
end

####
#### Checkpointer
####

mutable struct Checkpointer <: OutputWriter
   frequency :: Int
    interval :: FT
         dir :: String
      prefix :: String
     structs :: S
   fieldsets :: F
       force :: Bool
end

function Checkpointer(model; frequency=nothing, interval=nothing, dir=".", prefix="checkpoint",
                      structs = (:arch, :boundary_conditions, :grid, :clock, :eos, :constants, :closure),
                      fieldsets = (:velocities, :tracers, :G, :Gp),
                      force=false)

    frequency, interval = validate_interval(frequency, interval)

    if :forcing ∈ structs
        @error "Cannot checkpoint forcing functions :("
    end

    mkpath(dir)
    return Checkpointer(frequency, interval, dir, prefix, structs, fieldsets, force)
end

function savesubfields!(file, model, name, flds=propertynames(getproperty(model, name)))
    for f in flds
        file["$name/$f"] = Array(getproperty(getproperty(model, name), f).data.parent)
    end
    return
end

checkpointed_structs   = [:arch, :boundary_conditions, :grid, :clock, :eos, :constants, :closure]
checkpointed_fieldsets = [:velocities, :tracers, :G, :Gp]

function write_output(model, c::Checkpointer)
    @warn "Checkpointer will not save forcing functions, output writers, or diagnostics. They will need to be " *
          "restored manually."

    filepath = joinpath(c.dir, c.prefix * string(model.clock.iteration) * ".jld2")

    jldopen(filepath, "w") do file
        # Checkpointing model properties that we can just serialize.
        [file["$p"] = getproperty(model, p) for p in checkpointed_structs]

        # Checkpointing structs containing fields.
        [savesubfields!(file, model, p) for p in checkpointed_fieldsets]
    end
end

_arr(::CPU, a) = a
_arr(::GPU, a) = CuArray(a)

function restore_from_checkpoint(filepath)
    @warn "Checkpointer cannot restore forcing functions, output writers, or diagnostics. They will need to be " *
          "restored manually."

    kwargs = Dict{Symbol, Any}()  # We'll store all the kwargs we need to initialize a Model.

    file = jldopen(filepath, "r")

    # Restore model properties that were just serialized.
    for p in checkpointed_structs
        kwargs[Symbol(p)] = file["$p"]
    end

    # The Model constructor needs N and L.
    kwargs[:N] = (kwargs[:grid].Nx, kwargs[:grid].Ny, kwargs[:grid].Nz)
    kwargs[:L] = (kwargs[:grid].Lx, kwargs[:grid].Ly, kwargs[:grid].Lz)

    model =  Model(; kwargs...)

    for p in checkpointed_fieldsets
        for subp in propertynames(getproperty(model, p))
            getproperty(getproperty(model, p), subp).data.parent .= _arr(model.arch, file["$p/$subp"])
        end
    end

    close(file)

    return model
end

####
#### Output utils
####

"""
    HorizontalAverages(arch, grid)

Instantiate a struct of 1D arrays on `arch` and `grid`
for storing the results of horizontal averages of 3D fields.
"""
struct HorizontalAverages{A}
    U :: A
    V :: A
    T :: A
    S :: A
end

function HorizontalAverages(arch::CPU, grid::Grid{FT}) where FT
    U = zeros(FT, 1, 1, grid.Tz)
    V = zeros(FT, 1, 1, grid.Tz)
    T = zeros(FT, 1, 1, grid.Tz)
    S = zeros(FT, 1, 1, grid.Tz)

    HorizontalAverages(U, V, T, S)
end

function HorizontalAverages(arch::GPU, grid::Grid{FT}) where FT
    U = CuArray{FT}(undef, 1, 1, grid.Tz)
    V = CuArray{FT}(undef, 1, 1, grid.Tz)
    T = CuArray{FT}(undef, 1, 1, grid.Tz)
    S = CuArray{FT}(undef, 1, 1, grid.Tz)

    HorizontalAverages(U, V, T, S)
end

HorizontalAverages(model) = HorizontalAverages(model.arch, model.grid)

"""
    VerticalPlanes(arch, grid)

Instantiate a struct of 2D arrays on `arch` and `grid`
for storing the results of y-averages of 3D fields.
"""
struct VerticalPlanes{A}
    U :: A
    V :: A
    T :: A
    S :: A
end

function VerticalPlanes(arch::CPU, grid::Grid{FT}) where FT
    U = zeros(FT, grid.Tx, 1, grid.Tz)
    V = zeros(FT, grid.Tx, 1, grid.Tz)
    T = zeros(FT, grid.Tx, 1, grid.Tz)
    S = zeros(FT, grid.Tx, 1, grid.Tz)

    VerticalPlanes(U, V, T, S)
end

function VerticalPlanes(arch::GPU, grid::Grid{FT}) where FT
    U = CuArray{FT}(undef, grid.Tx, 1, grid.Tz)
    V = CuArray{FT}(undef, grid.Tx, 1, grid.Tz)
    T = CuArray{FT}(undef, grid.Tx, 1, grid.Tz)
    S = CuArray{FT}(undef, grid.Tx, 1, grid.Tz)

    VerticalPlanes(U, V, T, S)
end

VerticalPlanes(model) = VerticalPlanes(model.arch, model.grid)
