import JLD

"A type for writing checkpoints."
struct Checkpointer <: OutputWriter
    dir::AbstractString
    prefix::AbstractString
    frequency::Int
    padding::Int
end

"A type for writing NetCDF output."
mutable struct NetCDFOutputWriter <: OutputWriter
    dir::AbstractString
    prefix::AbstractString
    frequency::Int
    padding::Int
    compression::Int
end

"A type for writing Binary output."
mutable struct BinaryOutputWriter <: OutputWriter
    dir::AbstractString
    prefix::AbstractString
    frequency::Int
    padding::Int
end

function NetCDFOutputWriter(; dir=".", prefix="", frequency=1, padding=9, compression=5)
    NetCDFOutputWriter(dir, prefix, frequency, padding, compression)
end

"Return the filename suffix for the `OutputWriter` filetype."
suffix(fw::OutputWriter) = throw("Not implemented.")
suffix(fw::NetCDFOutputWriter) = ".nc"
suffix(fw::Checkpointer) = ".jld"

prefix(fw) = fw.prefix == "" ? "" : fw.prefix * "_"
filename(fw, name, time_step) = prefix(fw) * name * "_" * lpad(time_step, fw.padding, "0") * suffix(fw)
filename(fw::Checkpointer, name, time_step) = filename(fw, "model_checkpoint", time_step)

#
# Checkpointing functions
#

function write_output(model::Model, chk::Checkpointer)
    filepath = joinpath(chk.dir, filename(chk, model.clock.time_step))

    # Do not include the spectral solver parameters. We want to avoid serializing
    # FFTW and CuFFT plans as serializing functions is not supported by JLD, and
    # seems like a tricky business in general.
    model.ssp = nothing

    println("[Checkpointer] Serializing model to disk: $filepath")
    f = JLD.jldopen(filepath, "w", compress=true)
    JLD.@write f model
    close(f)

    # Reconstruct SpectralSolverParameters struct with FFT plans ?
    metadata, grid, stepper_tmp = model.metadata, model.grid, model.stepper_tmp
    if metadata.arch == :cpu
        stepper_tmp.fCC1.data .= rand(metadata.float_type, grid.Nx, grid.Ny, grid.Nz)
        ssp = SpectralSolverParameters(grid, stepper_tmp.fCC1, FFTW.PATIENT; verbose=true)
    elseif metadata.arch == :gpu
        stepper_tmp.fCC1.data .= CuArray{Complex{Float64}}(rand(metadata.float_type, grid.Nx, grid.Ny, grid.Nz))
        ssp = SpectralSolverParametersGPU(grid, stepper_tmp.fCC1)
    end
    return nothing 
end

function restore_from_checkpoint(filepath)
    println("Deserializing model from disk: $filepath")
    f = JLD.jldopen(filepath, "r")
    model = read(f, "model");
    close(f)

    # Reconstruct SpectralSolverParameters struct with FFT plans.
    metadata, grid, stepper_tmp = model.metadata, model.grid, model.stepper_tmp
    if metadata.arch == :cpu
        stepper_tmp.fCC1.data .= rand(metadata.float_type, grid.Nx, grid.Ny, grid.Nz)
        ssp = SpectralSolverParameters(grid, stepper_tmp.fCC1, FFTW.PATIENT; verbose=true)
    elseif metadata.arch == :gpu
        stepper_tmp.fCC1.data .= CuArray{Complex{Float64}}(rand(metadata.float_type, grid.Nx, grid.Ny, grid.Nz))
        ssp = SpectralSolverParametersGPU(grid, stepper_tmp.fCC1)
    end

    return model
end


#
# Binary output function
#

function write_output(model::Model, fw::BinaryOutputWriter)
    for (field, field_name) in zip(fw.fields, fw.field_names)
        filepath = joinpath(fw.dir, filename(fw, field_name, model.clock_time_step))

        println("[OutputWriter] Writing $field_name to disk: $filepath")
        if model.metadata == :cpu
            write(filepath, field.data)
        elseif model.metadata == :gpu
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

    xC = collect(model.grid.xC)
    yC = collect(model.grid.yC)
    zC = collect(model.grid.zC)

    xF = collect(model.grid.xF)
    yF = collect(model.grid.yF)
    zF = collect(model.grid.zF)

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

    filepath = joinpath(fw.dir, filename(fw, "", model.clock.time_step))

    (isfile(filepath) && overwrite) && rm(filepath)

    nccreate(filepath, "u", "xF", xF, xF_attr, 
                               "yC", yC, yC_attr, 
                               "zC", zC, zC_attr, 
                               atts=u_attr, compress=fw.compression)

    nccreate(filepath, "v", "xC", xC, xC_attr, 
                               "yF", yF, yC_attr, 
                               "zC", zC, zC_attr, 
                               atts=v_attr, compress=fw.compression)


    nccreate(filepath, "w", "xC", xC, xC_attr, 
                               "yC", yC, yC_attr, 
                               "zF", zF, zF_attr, 
                               atts=w_attr, compress=fw.compression)

    nccreate(filepath, "T", "xC", xC, xC_attr, 
                               "yC", yC, yC_attr, 
                               "zC", zC, zC_attr, 
                               atts=T_attr, compress=fw.compression)

    ncwrite(model.velocities.u.data, filepath, "u")
    ncwrite(model.velocities.v.data, filepath, "v")
    ncwrite(model.velocities.w.data, filepath, "w")
    ncwrite(model.tracers.T.data, filepath, "T")

    ncclose(filepath)

    return nothing 
end


