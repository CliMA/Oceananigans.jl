using Distributed
using NetCDF, JLD2

ext(fw::OutputWriter) = throw("Extension for $(typeof(fw)) is not implemented.")

function time_to_write(clock::Clock, diag::OutputWriter)
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

# When saving stuff to disk like a JLD2 file, `saveproperty!` is used, which
# converts Julia objects to language-agnostic objects.
saveproperty!(file, location, p::Number)        = file[location] = p
saveproperty!(file, location, p::AbstractRange) = file[location] = collect(p)
saveproperty!(file, location, p::AbstractArray) = file[location] = Array(p)
saveproperty!(file, location, p::Field)         = file[location] = Array(p.data.parent)
saveproperty!(file, location, p::Function) = @warn "Cannot save Function property into $location"
saveproperty!(file, location, p) = [saveproperty!(file, location * "/$subp", getproperty(p, subp)) for subp in propertynames(p)]

_bc_type(::BoundaryCondition{T}) where T = T

function saveproperty!(file, location, cbcs::CoordinateBoundaryConditions)
    for endpoint in propertynames(cbcs)
        endpoint_bc = getproperty(cbcs, endpoint)
        if isa(endpoint_bc.condition, Function)
            @warn "$field.$coord.$endpoint boundary is of type Function and cannot be saved to disk!"
            file["boundary/conditions/$field/$coord/$endpoint/type"] = string(_bc_type(endpoint_bc))
            file["boundary_conditions/$field/$coord/$endpoint/condition"] = missing
        else
            file["boundary_conditions/$field/$coord/$endpoint/type"] = string(_bc_type(endpoint_bc))
            file["boundary_conditions/$field/$coord/$endpoint/condition"] = endpoint_bc.condition
        end
    end
end

saveproperties!(file, structure, ps) = [saveproperty!(file, "$p", getproperty(structure, p)) for p in ps]

# When checkpointing, `serializeproperty!` is used, which serializes objects
# unless they need to be converted (basically CuArrays only).
serializeproperty!(file, location, p)        = file[location] = p
serializeproperty!(file, location, p::Field) = file[location] = Array(p.data.parent)
serializeproperty!(file, location, p::Function) = @warn "Cannot serialize Function property into $location"
serializeproperty!(file, location, p::Union{NamedTuple,AdamsBashforthTimestepper}) =
    [serializeproperty!(file, location * "/$subp", getproperty(p, subp)) for subp in propertynames(p)]

serializeproperties!(file, structure, ps) = [serializeproperty!(file, "$p", getproperty(structure, p)) for p in ps]

hasfunction(::AbstractArray{<:Number}) = false

function hasfunction(obj)
    if applicable(propertynames, obj) && length(propertynames(obj)) > 0
        return all([hasfunction(getproperty(obj, p)) for p in propertynames(obj)])
    elseif applicable(iterate, obj) && length(obj) > 1
        return all([hasfunction(elem) for elem in obj])
    else
        return isa(obj, Function)
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
        saveproperties!(file, model, including)
    end

    return JLD2OutputWriter(filepath, outputs, interval, frequency, init, including,
                            0.0, part, max_filesize, async, force, verbose)
end

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

    return
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
        saveproperties!(file, model, fw.including)
    end
end

function jld2output!(path, iter, time, data)
    jldopen(path, "r+") do file
        file["timeseries/t/$iter"] = time
        for (name, datum) in data
            file["timeseries/$name/$iter"] = datum
        end
    end
    return
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

mutable struct Checkpointer{I, T, P} <: OutputWriter
   frequency :: I
    interval :: T
         dir :: String
      prefix :: String
  properties :: P
       force :: Bool
end

function Checkpointer(model; frequency=nothing, interval=nothing, dir=".", prefix="checkpoint", force=false,
                      properties = [:arch, :boundary_conditions, :grid, :clock, :eos, :constants, :closure,
                                    :velocities, :tracers, :timestepper])

    frequency, interval = validate_interval(frequency, interval)

    if :grid ∉ properties
        @error ":grid not found in properties. The grid must be serialized for restore_from_checkpoint to work."
    end

    for p in properties
        isa(p, Symbol) || @error "Property $p to be checkpointed must be a Symbol."
        p ∉ propertynames(model) && @error "Cannot checkpoint $p, it is not a model property!"
    end

    if :boundary_conditions ∈ properties && hasfunction(model.boundary_conditions)
        @warn "One or more boundary conditions contain functions and will not be checkpointed."
        filter!(e -> e != :boundary_conditions, properties)
    end

    mkpath(dir)
    return Checkpointer(frequency, interval, dir, prefix, properties, force)
end

function write_output(model, c::Checkpointer)
    filepath = joinpath(c.dir, c.prefix * string(model.clock.iteration) * ".jld2")
    jldopen(filepath, "w") do file
        file["checkpointed_properties"] = c.properties

        # Serialize boundary conditions whole as serializeproperty! will attempt to
        # recurse through them as it's a named tuple.
        if :boundary_conditions ∈ c.properties
            file["boundary_conditions"] = model.boundary_conditions
        end

        serializeproperties!(file, model, filter(e -> e != :boundary_conditions, c.properties))
    end
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

const field_containing_structs = (:velocities, :tracers, :timestepper)

function restore_from_checkpoint(filepath; kwargs = Dict{Symbol, Any}())
    file = jldopen(filepath, "r")

    cps = file["checkpointed_properties"]

    # Restore model properties that were just serialized.
    # We skip fields that contain structs and restore them after model creation.
    for p in cps
        if p ∉ field_containing_structs
            kwargs[p] = file["$p"]
        end
    end

    # The Model constructor needs N and L.
    kwargs[:N] = (kwargs[:grid].Nx, kwargs[:grid].Ny, kwargs[:grid].Nz)
    kwargs[:L] = (kwargs[:grid].Lx, kwargs[:grid].Ly, kwargs[:grid].Lz)

    model =  Model(; kwargs...)

    # Now restore fields.
    for p in cps
        if p in field_containing_structs
            restore_fields!(model, file, model.arch, p)
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
