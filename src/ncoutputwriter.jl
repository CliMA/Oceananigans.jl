"""
    WriteGeometry(model; slice_kw...)

Writes a geometry.nc file that contains all the dimensions of the domain.
"""
function WriteGeometry(model; filename="./geometry.nc", slice_kw...)
    dimensions = Dict(
        "xC" => collect(model.grid.xC),
        "yC" => collect(model.grid.yC),
        "zC" => collect(model.grid.zC),
        "xF" => collect(model.grid.xF),
        "yF" => collect(model.grid.yF),
        "zF" => collect(model.grid.zF)
    )

    for (d, slice) in slice_kw
        if String(d) in keys(dimensions)
            dimensions[String(d)] = dimensions[String(d)][slice]
        end
    end

    Dataset(filename, "c") do ds
        for (dimname, dimarray) in dimensions
            defDim(ds, dimname, length(dimarray)); sync(ds)
            defVar(ds, dimname, dimarray, (dimname,))
        end
    end
end

function NCOutputWriterInit(model; filename="output.nc", slice_kw...)
    WriteGeometry(model; filename=filename, slice_kw...)
end

function slice(field; slice_kw...)
    slice = Vector{Union{AbstractRange,Colon}}()
    for dim in dims(field)
        Hxyorz =  getproperty(field.grid, Symbol("H"*dim[1]))
        Nxyorz =  getproperty(field.grid, Symbol("N"*dim[1]))
        push!(slice, haskey(slice_kw, Symbol(dim)) ? slice_kw[Symbol(dim)] : ((Hxyorz+1):(Nxyorz+1)))
    end
    @show(slice)
    getindex(field.data.parent, slice...)
end

#function NCOutputWriter(model, fields, interval; filename="output.nc", slice_kw...)
function NCOutputWriter(model, fields; filename="output.nc", slice_kw...)
    NCOutputWriterInit(model; filename=filename, slice_kw...)
    Dataset(filename,"a") do ds
        for (fieldname, field) in fields
            defVar(ds, fieldname, slice(field; slice_kw...), dims(field))
        end
    end
end


# """
#     NCOutputWriter{F, I, O, IF, IN, KW} <: AbstractOutputWriter

# An improved output writer for writing to NetCDF files.
# """
# mutable struct NCOutputWriter{F, I, O, IF, IN, KW} <: AbstractOutputWriter
#         filepath :: String
#           prefix :: String
#          outputs :: O
#         interval :: I
#        frequency :: F
#             init :: IF
#        including :: IN
#          clobber :: Bool
# end

# noinit(args...) = nothing

# """
#     NCOutputWriter(model, outputs; interval=nothing, frequency=nothing, dir=".",
#                      prefix="", init=noinit, including=[:grid, :coriolis, :buoyancy, :closure],
#                      part=1, max_filesize=Inf, clobber=false, async=false, verbose=false)

# Construct a `NCOutputWriter` that writes `label, func` pairs in `outputs` (which can be a `Dict` or `NamedTuple`)
# to a NC file, where `label` is a symbol that labels the output and `func` is a function of the form `func(model)`
# that returns the data to be saved.

# Keyword arguments
# =================

#     - `frequency::Int`   : Save output every `n` model iterations.
#     - `interval::Int`    : Save output every `t` units of model clock time.
#     - `filepath::String` : Directory to save output to. Default: "." (current working directory).
#     - `prefix::String`   : A prefix that is added to the saved filename. Default: "".
#     - `clobber::Bool`    : Remove existing files if their filenames conflict. Default: `false`.
#     - `async::Bool`      : Write output asynchronously. Default: `false`.
# """

# function NCOutputWriter(model, outputs; interval=nothing, frequency=nothing, filepath=".",
#                         prefix="", clobber=false, async=false)

#     validate_interval(interval, frequency)

#     mkpath(dir)
#     filepath = joinpath(filepath, prefix * ".nc")
#     mode = clobber ? "c" : "a"

#     Dataset(filepath, mode, attrib = [init(file, model)]) do ds
#         write_output(model, file)
#     end

#     return NCOutputWriter(filepath, outputs, interval, frequency, async, clobber)
# end


# function write_output(model::Model, fw::NetCDFOutputWriter)
#     fields = Dict(
#         "xC" => collect(model.grid.xC),
#         "yC" => collect(model.grid.yC),
#         "zC" => collect(model.grid.zC),
#         "xF" => collect(model.grid.xF),
#         "yF" => collect(model.grid.yF),
#         "zF" => collect(model.grid.zF),
#         "u" => Array(parentdata(model.velocities.u)),
#         "v" => Array(parentdata(model.velocities.v)),
#         "w" => Array(parentdata(model.velocities.w)),
#         "T" => Array(parentdata(model.tracers.T)),
#         "S" => Array(parentdata(model.tracers.S))
#     )

#     if fw.async
#         # Execute asynchronously on worker 2.
#         i = model.clock.iteration
#         @async remotecall(write_output_netcdf, 2, fw, fields, i)
#     else
#         write_output_netcdf(fw, fields, model.clock.iteration)
#     end

#     return
# end

# function write_output_netcdf(fw::NetCDFOutputWriter, fields, iteration)
#     xC, yC, zC = fields["xC"], fields["yC"], fields["zC"]
#     xF, yF, zF = fields["xF"], fields["yF"], fields["zF"]

#     u, v, w = fields["u"], fields["v"], fields["w"]
#     T, S    = fields["T"], fields["S"]

#     xC_attr = Dict("longname" => "Locations of the cell centers in the x-direction.", "units" => "m")
#     yC_attr = Dict("longname" => "Locations of the cell centers in the y-direction.", "units" => "m")
#     zC_attr = Dict("longname" => "Locations of the cell centers in the z-direction.", "units" => "m")

#     xF_attr = Dict("longname" => "Locations of the cell faces in the x-direction.", "units" => "m")
#     yF_attr = Dict("longname" => "Locations of the cell faces in the y-direction.", "units" => "m")
#     zF_attr = Dict("longname" => "Locations of the cell faces in the z-direction.", "units" => "m")

#     u_attr = Dict("longname" => "Velocity in the x-direction", "units" => "m/s")
#     v_attr = Dict("longname" => "Velocity in the y-direction", "units" => "m/s")
#     w_attr = Dict("longname" => "Velocity in the z-direction", "units" => "m/s")
#     T_attr = Dict("longname" => "Temperature", "units" => "K")
#     S_attr = Dict("longname" => "Salinity", "units" => "g/kg")

#     filepath = joinpath(fw.dir, filename(fw, "", iteration))

#     if fw.async
#         println("[Worker $(Distributed.myid()): NetCDFOutputWriter] Writing fields to disk: $filepath")
#     end

#     isfile(filepath) && rm(filepath)

#     nccreate(filepath, "u", "xF", xC, xC_attr,
#                             "yC", yC, yC_attr,
#                             "zC", zC, zC_attr,
#                             atts=u_attr, compress=fw.compression)

#     nccreate(filepath, "v", "xC", xC, xC_attr,
#                             "yF", yC, yC_attr,
#                             "zC", zC, zC_attr,
#                             atts=v_attr, compress=fw.compression)

#     nccreate(filepath, "w", "xC", xC, xC_attr,
#                             "yC", yC, yC_attr,
#                             "zF", zC, zC_attr,
#                             atts=w_attr, compress=fw.compression)

#     nccreate(filepath, "T", "xC", xC, xC_attr,
#                             "yC", yC, yC_attr,
#                             "zC", zC, zC_attr,
#                             atts=T_attr, compress=fw.compression)

#     nccreate(filepath, "S", "xC", xC, xC_attr,
#                             "yC", yC, yC_attr,
#                             "zC", zC, zC_attr,
#                             atts=S_attr, compress=fw.compression)

#     ncwrite(u, filepath, "u")
#     ncwrite(v, filepath, "v")
#     ncwrite(w, filepath, "w")
#     ncwrite(T, filepath, "T")
#     ncwrite(S, filepath, "S")

#     ncclose(filepath)

#     return
# end
