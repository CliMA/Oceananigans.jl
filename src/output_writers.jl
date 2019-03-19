import JLD
using Distributed

using NetCDF
@everywhere using NetCDF

@everywhere abstract type OutputWriter end

"A type for writing checkpoints."
struct Checkpointer <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
    padding::Int
end

"A type for writing NetCDF output."
mutable struct NetCDFOutputWriter <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
    padding::Int
    compression::Int
    async::Bool
end

# Not sure why I have to define NetCDFOutputWriter again @everywhere but I get a weird ArgumentError
# if I just define it @everywhere. I have to define it twice...
@everywhere begin
    mutable struct NetCDFOutputWriter <: OutputWriter
        dir::AbstractString
        filename_prefix::AbstractString
        output_frequency::Int
        padding::Int
        compression::Int
        async::Bool
    end
end

"A type for writing Binary output."
mutable struct BinaryOutputWriter <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
    padding::Int
end

function NetCDFOutputWriter(; dir=".", prefix="", frequency=1, padding=9, compression=5, async=false)
    NetCDFOutputWriter(dir, prefix, frequency, padding, compression, async)
end

"Return the filename extension for the `OutputWriter` filetype."
ext(fw::OutputWriter) = throw("Not implemented.")
ext(fw::NetCDFOutputWriter) = ".nc"
@everywhere ext(fw::NetCDFOutputWriter) = ".nc"
ext(fw::Checkpointer) = ".jld"

filename(fw, name, iteration) = fw.filename_prefix * name * lpad(iteration, fw.padding, "0") * ext(fw)
@everywhere filename(fw, name, iteration) = fw.filename_prefix * name * lpad(iteration, fw.padding, "0") * ext(fw)
filename(fw::Checkpointer, name, iteration) = filename(fw, "model_checkpoint", iteration)

#
# Checkpointing functions
#

function write_output(model::Model, chk::Checkpointer)
    filepath = joinpath(chk.dir, filename(chk, model.clock.iteration))

    # Do not include the spectral solver parameters. We want to avoid serializing
    # FFTW and CuFFT plans as serializing functions is not supported by JLD, and
    # seems like a tricky business in general.
    model.poisson_solver = nothing

    println("[Checkpointer] Serializing model to disk: $filepath")
    f = JLD.jldopen(filepath, "w", compress=true)
    JLD.@write f model
    close(f)

    # Reconstruct PoissonSolver struct with FFT plans ?
    metadata, grid, stepper_tmp = model.metadata, model.grid, model.stepper_tmp
    if metadata.arch == :cpu
        stepper_tmp.fCC1.data .= rand(metadata.float_type, grid.Nx, grid.Ny, grid.Nz)
        poisson_solver = PoissonSolver(grid, stepper_tmp.fCC1, FFTW.PATIENT; verbose=true)
    elseif metadata.arch == :gpu
        stepper_tmp.fCC1.data .= CuArray{Complex{Float64}}(rand(metadata.float_type, grid.Nx, grid.Ny, grid.Nz))
        poisson_solver = PoissonSolverGPU(grid, stepper_tmp.fCC1)
    end
    return nothing
end

function restore_from_checkpoint(filepath)
    println("Deserializing model from disk: $filepath")
    f = JLD.jldopen(filepath, "r")
    model = read(f, "model");
    close(f)

    # Reconstruct PoissonSolver struct with FFT plans.
    metadata, grid, stepper_tmp = model.metadata, model.grid, model.stepper_tmp
    if metadata.arch == :cpu
        stepper_tmp.fCC1.data .= rand(metadata.float_type, grid.Nx, grid.Ny, grid.Nz)
        poisson_solver = PoissonSolver(grid, stepper_tmp.fCC1, FFTW.PATIENT; verbose=true)
    elseif metadata.arch == :gpu
        stepper_tmp.fCC1.data .= CuArray{Complex{Float64}}(rand(metadata.float_type, grid.Nx, grid.Ny, grid.Nz))
        poisson_solver = PoissonSolverGPU(grid, stepper_tmp.fCC1)
    end

    return model
end


#
# Binary output function
#

function write_output(model::Model, fw::BinaryOutputWriter)
    for (field, field_name) in zip(fw.fields, fw.field_names)
        filepath = joinpath(fw.dir, filename(fw, field_name, model.clock.iteration))

        println("[BinaryOutputWriter] Writing $field_name to disk: $filepath")
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
    fields = Dict(
        "xC" => collect(model.grid.xC),
        "yC" => collect(model.grid.yC),
        "zC" => collect(model.grid.zC),
        "xF" => collect(model.grid.xF),
        "yF" => collect(model.grid.yF),
        "zF" => collect(model.grid.zF),
        "u" => Array(model.velocities.u.data),
        "v" => Array(model.velocities.v.data),
        "w" => Array(model.velocities.w.data),
        "T" => Array(model.tracers.T.data),
        "S" => Array(model.tracers.S.data)
    )
    
    if fw.async
        # Execute asynchronously on worker 2.
        println("Using @async...")
        println("nprocs()=$(nprocs())")
        @async remotecall(write_output_netcdf, 2, fw, fields, model.clock.iteration)
    else
        println("Regular call...")
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

# Not sure why I have to define write_output_netcdf again @everywhere but if I just define it @everywhere
# then I get UndefVarError: write_output_netcdf not defined...
@everywhere function write_output_netcdf(fw::NetCDFOutputWriter, fields, iteration)
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
