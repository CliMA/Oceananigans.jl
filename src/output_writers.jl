using Serialization

struct SavedFields
    u::Array{Float64,4}
    w::Array{Float64,4}
    T::Array{Float64,4}
    ρ::Array{Float64,4}
    ΔR  # Output frequency
end

function SavedFields(g, Nt, ΔR)
    u = zeros(Int(Nt/ΔR), g.Nx, g.Ny, g.Nz)
    w = zeros(Int(Nt/ΔR), g.Nx, g.Ny, g.Nz)
    T = zeros(Int(Nt/ΔR), g.Nx, g.Ny, g.Nz)
    ρ = zeros(Int(Nt/ΔR), g.Nx, g.Ny, g.Nz)
    SavedFields(u, w, T, ρ, ΔR)
end

struct Checkpointer <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
end

function write_output(model::Model, chk::Checkpointer)
    filename = chk.filename_prefix * "_model_checkpoint_" * lpad(model.clock.time, 12, "0") * ".jlser"
    filepath = joinpath(chk.dir, filename)

    println("[Checkpointer] Serializing model to disk: $filepath")
    serialize(filepath, model)
end

function read_output(filepath)
    println("Deserializing model from disk: $filepath")
    deserialize(filepath)
end

struct FieldWriter <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
    fields::Array{Field,1}
    field_names::Array{String,1}
end

function write_output(model::Model, fw::FieldWriter)
    for (field, field_name) in zip(fw.fields, fw.field_names)
        filename = fw.filename_prefix * "_" * field_name * "_" * lpad(model.clock.time, 12, "0") * ".dat"
        filepath = joinpath(fw.dir, filename)

        println("[FieldWriter] Writing $field_name to disk: $filepath")
        write(filepath, field.data)
    end
end

function read_output(model::Model, fw::FieldWriter, field_name, time)
    filename = fw.filename_prefix * "_" * field_name * "_" * lpad(time, 12, "0") * ".dat"
    filepath = joinpath(fw.dir, filename)

    fio = open(filepath, "r")
    arr = zeros(model.metadata.float_type, model.grid.Nx, model.grid.Ny, model.grid.Nz)
    read!(fio, arr)
    return arr
end
