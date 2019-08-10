using Distributed

using NetCDF

"A type for writing NetCDF output."
mutable struct NetCDFOutputWriter <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
    padding::Int
    naming_scheme::Symbol
    compression::Int
    async::Bool
end

"A type for writing Binary output."
mutable struct BinaryOutputWriter <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
    padding::Int
end

function NetCDFOutputWriter(; dir=".", prefix="", frequency=1, padding=9,
                              naming_scheme=:iteration, compression=3, async=false)
    NetCDFOutputWriter(dir, prefix, frequency, padding, naming_scheme, compression, async)
end

"Return the filename extension for the `OutputWriter` filetype."
ext(fw::OutputWriter) = throw("Not implemented.")
ext(fw::NetCDFOutputWriter) = ".nc"

function filename(fw, name, iteration)
    if fw.naming_scheme == :iteration
        fw.filename_prefix * name * lpad(iteration, fw.padding, "0") * ext(fw)
    elseif fw.naming_scheme == :file_number
        file_num = Int(iteration / fw.output_frequency)
        fw.filename_prefix * name * lpad(file_num, fw.padding, "0") * ext(fw)
    else
        throw(ArgumentError("Invalid naming scheme: $(fw.naming_scheme)"))
    end
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

#
# NetCDF output functions
#
# Eventually, we want to permit the user to flexibly define what is outputted.
# The user-defined function that produces output may involve computations, launching kernels,
# etc; so this API needs to be designed. For now, we simply save u, v, w, and T.

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
    else
        println("[NetCDFOutputWriter] Writing fields to disk: $filepath")
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
    println("[NetCDFOutputWriter] Reading fields from disk: $filepath")
    field_data = ncread(filepath, field_name)
    ncclose(filepath)
    return field_data
end

"""
    JLD2OutputWriter(model, outputs; dir=".", prefix="", interval=1, init=noinit,
                     force=false, asynchronous=false)

Construct an `OutputWriter` that writes `label, fcn` pairs in the
dictionary `outputs` to a single `JLD2` file, where `label` is a symbol that labels
the output and `fcn` is a function of the form `fcn(model)` that returns
the data to be saved. The keyword `init` is a function of the form `init(file, model)`
that runs when the JLD2 output file is initialized.
"""
mutable struct JLD2OutputWriter{F, I, O} <: OutputWriter
        filepath :: String
         outputs :: O
        interval :: I
       frequency :: F
        previous :: Float64
    asynchronous :: Bool
    verbose      :: Bool
end

noinit(args...) = nothing

function JLD2OutputWriter(model, outputs; interval=nothing, frequency=nothing, dir=".", prefix="",
                          init=noinit, including=[:grid, :eos, :constants, :closure],
                          force=false, asynchronous=false, verbose=false)

    interval === nothing && frequency === nothing &&
        error("Either interval or frequency must be passed to the JLD2OutputWriter!")

    mkpath(dir)
    filepath = joinpath(dir, prefix * ".jld2")
    force && isfile(filepath) && rm(filepath, force=true)

    jldopen(filepath, "a+") do file
        init(file, model)
        savesubstructs!(file, model, including)
    end

    return JLD2OutputWriter(filepath, outputs, interval, frequency, 0.0, asynchronous, verbose)
end

function savesubstruct!(file, model, name, flds=propertynames(getproperty(model, name)))
    for fld in flds
        file["$name/$fld"] = getproperty(getproperty(model, name), fld)
    end
    return nothing
end

savesubstructs!(file, model, names) = [savesubstruct!(file, model, name) for name in names]

function write_output(model, fw::JLD2OutputWriter)
    fw.verbose && @info @sprintf("Calculating JLD2 output %s...", keys(fw.outputs))
    t0 = @elapsed data = Dict((name, f(model)) for (name, f) in fw.outputs)
    fw.verbose && @info "Calculation time: $(prettytime(time_ns() - t0))"

    iter = model.clock.iteration
    time = model.clock.time
    path = fw.filepath

    fw.verbose && @info @sprintf("Writing JLD2 output %s...", keys(fw.outputs))
    t0 = time_ns()
    if fw.asynchronous
        @async remotecall(jld2output!, 2, path, iter, time, data)
    else
        jld2output!(path, iter, time, data)
    end
    fw.verbose && @info "Writing time: $(prettytime(time_ns()-t0)))"

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

#
# Output utils
#

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
