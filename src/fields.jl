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

struct Fields{T <: AbstractFloat} <: FieldCollection
    u::FaceField
    v::FaceField
    w::FaceField
    T::ZoneField
    S::ZoneField
    ρ::ZoneField
    p::ZoneField
    pHY::ZoneField
    pHY′::ZoneField
    pNHS::ZoneField
end

function Fields(g::RegularCartesianGrid, T=Float64)
    u = FaceField(g, T)
    v = FaceField(g, T)
    w = FaceField(g, T)
    T = ZoneField(g, T)
    S = ZoneField(g, T)
    ρ = ZoneField(g, T)
    p = ZoneField(g, T)
    pHY = ZoneField(g, T)
    pHY′ = ZoneField(g, T)
    pNHS = ZoneField(g, T)
    Fields{T}(u, v, w, T, S, ρ, p, pHY, pHY′, pNHS)
end

struct SourceTermFields{T <: AbstractFloat} <: FieldCollection
    Gu::FaceField
    Gv::FaceField
    Gw::FaceField
    GT::ZoneField
    GS::ZoneField
end

function SourceTerms(g::RegularCartesianGrid, T=Float64)
    Gu = FaceField(g, T)
    Gv = FaceField(g, T)
    Gw = FaceField(g, T)
    GT = ZoneField(g, T)
    GS = ZoneField(g, T)
end
