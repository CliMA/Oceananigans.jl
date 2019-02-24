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
