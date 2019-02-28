import JLD

struct Checkpointer <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
end

function write_output(model::Model, chk::Checkpointer)
    filename = chk.filename_prefix * "_model_checkpoint_" * lpad(model.clock.time_step, 9, "0") * ".jld"
    filepath = joinpath(chk.dir, filename)

    # Do not include the spectral solver parameters. We want to avoid serializing
    # FFTW and CuFFT plans as serializing functions is not supported by JLD, and
    # seems like a tricky business in general.
    model.ssp = nothing

    println("[Checkpointer] Serializing model to disk: $filepath")
    f = JLD.jldopen(filepath, "w", compress=true)
    JLD.@write f model
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

struct BinaryFieldWriter <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
    fields::Array{Field,1}
    field_names::Array{String,1}
end

function write_output(model::Model, fw::BinaryFieldWriter)
    for (field, field_name) in zip(fw.fields, fw.field_names)
        filename = fw.filename_prefix * "_" * field_name * "_" * lpad(model.clock.time_step, 9, "0") * ".dat"
        filepath = joinpath(fw.dir, filename)

        println("[FieldWriter] Writing $field_name to disk: $filepath")
        if model.metadata == :cpu
            write(filepath, field.data)
        elseif model.metadata == :gpu
            write(filepath, Array(field.data))
        end
    end
end

function read_output(model::Model, fw::BinaryFieldWriter, field_name, time)
    filename = fw.filename_prefix * "_" * field_name * "_" * lpad(time_step, 9, "0") * ".dat"
    filepath = joinpath(fw.dir, filename)

    fio = open(filepath, "r")
    arr = zeros(model.metadata.float_type, model.grid.Nx, model.grid.Ny, model.grid.Nz)
    read!(fio, arr)
    return arr
end

struct NetCDFFieldWriter <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
end

function write_output(model::Model, fw::NetCDFFieldWriter)
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

    nc_filename = fw.filename_prefix * "_" * lpad(model.clock.time_step, 9, "0") * ".nc"
    nc_filepath = joinpath(fw.dir, nc_filename)

    isfile(nc_filepath) && rm(nc_filepath)

    nccreate(nc_filepath, "u", "xC", xC, xC_attr, "yC", yC, yC_attr, "zC", zC, zC_attr, atts=u_attr, compress=5)
    ncwrite(model.velocities.u.data, nc_filepath, "u")

    nccreate(nc_filepath, "v", "xC", xC, xC_attr, "yC", yC, yC_attr, "zC", zC, zC_attr, atts=v_attr, compress=5)
    ncwrite(model.velocities.v.data, nc_filepath, "v")

    nccreate(nc_filepath, "w", "xC", xC, xC_attr, "yC", yC, yC_attr, "zC", zC, zC_attr, atts=w_attr, compress=5)
    ncwrite(model.velocities.w.data, nc_filepath, "w")

    nccreate(nc_filepath, "T", "xC", xC, xC_attr, "yC", yC, yC_attr, "zC", zC, zC_attr, atts=T_attr, compress=5)
    ncwrite(model.tracers.T.data, nc_filepath, "T")

    ncclose(nc_filename)
end

function read_output(fw::NetCDFFieldWriter, field_name, iter)
    nc_filename = fw.filename_prefix * "_" * lpad(iter, 9, "0") * ".nc"
    nc_filepath = joinpath(fw.dir, nc_filename)
    ncread(nc_filepath, field_name)
end
