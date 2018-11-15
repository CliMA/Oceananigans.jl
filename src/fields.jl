struct ZoneField{T <: AbstractFloat} <: Field
    g::Grid
    f::Array
end

struct FaceField{T <: AbstractFloat} <: Field
    g::Grid
    f::Array
end

function ZoneField(g::RegularCartesianGrid, T=Float64)
    f = Array{T,g.dim}(undef, g.Nx, g.Ny, g.Nz)
    ZoneField{T}(g, f)
end

function FaceField(g::RegularCartesianGrid, T=Float64)
    f = Array{T,g.dim}(undef, g.Nx + 1, g.Ny + 1, g.Nz + 1)
    FaceField{T}(g, f)
end

struct VelocityFields{T <: AbstractFloat} <: FieldCollection
    u::FaceField
    v::FaceField
    w::FaceField
end

function VelocityFields(g::RegularCartesianGrid, T=Float64)
    u = FaceField(g, T)
    v = FaceField(g, T)
    w = FaceField(g, T)
end
