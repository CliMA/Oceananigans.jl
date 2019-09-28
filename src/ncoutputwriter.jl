"""
    WriteGeometry(model; filename="./geometry.nc", mode="c", slice_kw...)

Writes a geometry.nc file that contains all the dimensions of the domain.

Keyword arguments
=================

    - `filename::String`  : File name to be saved under
    - `mode::String`      : Netcdf file is opened in either clobber ("c") or append ("a") mode (Default: "c" )
"""
function WriteGeometry(model; filename="./geometry.nc", mode="c", slice_kw...)
    dimensions = Dict(
        "xC" => collect(model.grid.xC),
        "yC" => collect(model.grid.yC),
        "zC" => collect(model.grid.zC),
        "xF" => collect(model.grid.xF),
        "yF" => collect(model.grid.yF),
        "zF" => collect(model.grid.zF)
    )

    # Applies slices to the dimensions d
    for (d, slice) in slice_kw
        if String(d) in keys(dimensions)
            dimensions[String(d)] = dimensions[String(d)][slice]
        end
    end

    # Writes the sliced dimensions to the specified netcdf file
    Dataset(filename, mode) do ds
        for (dimname, dimarray) in dimensions
            defDim(ds, dimname, length(dimarray)); sync(ds)
            defVar(ds, dimname, dimarray, (dimname,))
        end
    end
end



mutable struct NCOutputWriter <: AbstractOutputWriter
        filename :: String
         dataset :: Any
         outputs :: Dict
        interval :: Union{Nothing, AbstractFloat}
       frequency :: Union{Nothing, Int}
         attribs :: Dict
         clobber :: Bool
          slices :: Dict
              nt :: Int
        previous :: Float64
end

"""
    NCOutputWriter(model, outputs; interval=nothing, frequency=nothing, filename=".",
                                   clobber=true, attribs=Dict(), slice_kw...)

Construct a `NCOutputWriter` that writes `label, func` pairs in `outputs` (which can be a `Dict` or `NamedTuple`)
to a NC file, where `label` is a symbol that labels the output and `func` is a function of the form `func(model)`
that returns the data to be saved.

Keyword arguments
=================

    - `filename::String` : Directory to save output to. Default: "." (current working directory).
    - `frequency::Int`   : Save output every `n` model iterations.
    - `interval::Int`    : Save output every `t` units of model clock time.
    - `clobber::Bool`    : Remove existing files if their filenames conflict. Default: `false`.
    - `attribs::Array`   : List of model properties to save with every file. By default, the
                           grid, equation of state, coriolis parameters, buoyancy parameters,
                           and turbulence closure parameters are saved.
    - `slice_kw`         : dimname = OrdinalRange will slice the dimension `dimnamee` according
                           to OrdinalRange
                           e.g. xC = 3:10 will only output the dimension `xC` between indices 3 and 10.
"""

function NCOutputWriter(model, outputs; interval=nothing, frequency=nothing, filename=".",
                        clobber=true, attribs=Dict(), slice_kw...)

    validate_interval(interval, frequency)

    mode = clobber ? "c" : "a"

    # Initiates the output file with dimensions
    WriteGeometry(model; filename=filename, mode=mode, slice_kw...)

    # Opens the same output file for writing fields from the user-supplied variable outputs
    dataset = Dataset(filename, "a")

    # Creates an unliimited dimension "Time"
    defDim(dataset, "Time", Inf); sync(dataset)
    defVar(dataset, "Time", Float32, ("Time",)); sync(dataset)
    nt = 0 # Number of time-steps

    # Initiates empty Float32 arrays for fields from the user-supplied variable outputs
    for (fieldname, field) in outputs
        defVar(dataset, fieldname, Float32, (dims(field)...,"Time")); sync(dataset)
    end

    # Stores slices for the dimensions of each output field
    slices = Dict{String, Vector{Union{OrdinalRange,Colon}}}()
    for (fieldname, field) in outputs
        slices[fieldname] = slice(field; slice_kw...)
    end

    return NCOutputWriter(filename, dataset, outputs, interval,
                          frequency, attribs, clobber, slices, nt, 0.0)
end

# Closes the outputwriter
function OWClose(fw::NCOutputWriter)
    close(fw.dataset)
end


"""
    slice(field; slice_kw...)

For internal use only. Returns a slice for a field based on its dimensions and the supplied slices in `slice_kw`.
"""
function slice(field; slice_kw...)
    slice = Vector{Union{AbstractRange,Colon}}()
    for dim in dims(field)
        # Hx, Hy, or Hz based on the dimension in consideration
        Hxyorz =  getproperty(field.grid, Symbol("H"*dim[1]))

        # Nx, Ny, or Nz based on the dimension in consideration
        Nxyorz =  getproperty(field.grid, Symbol("N"*dim[1]))

        # Creates a slice and stores it (TODO: This is more complicated because of halos, can be simplified)
        push!(slice, haskey(slice_kw, Symbol(dim)) ? slice_kw[Symbol(dim)] : ((Hxyorz+1):(Nxyorz+1)))
    end
    return slice
end

# Appends a dimension at the end of an array
add_dim(x::Array) = reshape(x, (size(x)...,1))

"""
    write_output(model, OutputWriter)

For internal user only. Writes output to the netcdf file at specified intervals. Increments the `Time` dimension every time an output is written to the file.
"""
function write_output(model, fw::NCOutputWriter)
    fw.nt += 1
    fw.dataset["Time"][fw.nt] = model.clock.time
    for (fieldname, field) in fw.outputs
        fw.dataset[fieldname][:,:,:,fw.nt] = add_dim(getindex(field.data.parent, fw.slices[fieldname]...))
    end
    sync(fw.dataset)
end
