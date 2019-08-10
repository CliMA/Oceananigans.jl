using Distributed
using NetCDF, JLD2

ext(fw::OutputWriter) = throw("Extension for $(typeof(fw)) is not implemented.")

####
#### Binary output writer
####

"A type for writing Binary output."
Base.@kwdef mutable struct BinaryOutputWriter <: OutputWriter
                 dir :: AbstractString = "."
              prefix :: AbstractString = ""
    output_frequency :: Int = 1
    padding          :: Int = 9
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
    output_frequency :: Int    = 1
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
        file_num = Int(iteration / fw.output_frequency)
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
    max_filesize :: Int
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
                          part=1, max_filesize=Inf, force=false, =false, verbose=false)

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
                            0.0, part, max_filesize, , force, verbose)
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
    tc = @elapsed data = Dict((name, f(model)) for (name, f) in fw.outputs)
    verbose && @info "Calculation time: $(prettytime(tc))"

    sz = filesize(fw.filepath)
    if sz > fw.max_filesize
        verbose && @info "Filesize $(pretty_filesize(sz)) has exceeded maximum file size $(pretty_filesize(fw.max_filesize))."

        if fw.part == 1
            part1_path = replace(fw.filepath, r".jld2$" => "_part1.jld2")
            verbose && @info "Renaming first part: $(fw.filepath) -> $part1_path"
            mv(fw.filepath, part1_path, force=fw.force)
            fw.filepath = part1_path
        end

        fw.part += 1
        fw.filepath = replace(fw.filepath, r"part\d+.jld2$" => "part" * string(fw.part) * ".jld2")
        verbose && @info "Now writing to: $(fw.filepath)"

        jldopen(fw.filepath, "a+") do file
            fw.init(file, model)
            savesubstructs!(file, model, fw.including)
        end
    end

    iter = model.clock.iteration
    time = model.clock.time
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

function jld2output!(path, iter, time, data)
    jldopen(path, "r+") do file
        file["timeseries/t/$iter"] = time
        for (name, datum) in data
            file["timeseries/$name/$iter"] = datum
        end
    end
    return nothing
end

time_to_write(clock, out::OutputWriter) = (clock.iteration % out.output_frequency) == 0
time_to_write(clock, out::JLD2OutputWriter) = (clock.iteration % out.frequency) == 0

function time_to_write(clock, out::JLD2OutputWriter{<:Nothing})
    if clock.time > out.previous + out.interval
        out.previous = clock.time - rem(clock.time, out.interval)
        return true
    else
        return false
    end
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
