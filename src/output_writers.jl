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

function deserialize_model(filepath)
    println("Deserializing model from disk: $filepath")
    deserialize(filepath)
end
