struct ZoneField{T <: AbstractFloat} <: Field
    f::Array
end

struct FaceField{T <: AbstractFloat} <: Field
    f::Array
end

function ZoneField(g::RegularCartesianGrid, T=Float64)
    f = Array{T,g.dim}(undef, g.Nx, g.Ny, g.Nz)
    ZoneField{T}(f)
end

function FaceField(g::RegularCartesianGrid, T=Float64)
    f = Array{T,g.dim}(undef, g.Nx + 1, g.Ny + 1, g.Nz + 1)
    FaceField{T}(f)
end

struct Fields{T <: AbstractFloat} <: FieldCollection
    u::FaceField{T}
    v::FaceField{T}
    w::FaceField{T}
    T::ZoneField{T}
    ρ::ZoneField{T}
    p::ZoneField{T}
    S::ZoneField{T}
    pHY::ZoneField{T}
    pHY′::ZoneField{T}
    pNHS::ZoneField{T}
end

function Fields(g::RegularCartesianGrid, T=Float64)
    u = FaceField(g, T)
    v = FaceField(g, T)
    w = FaceField(g, T)
    θ = ZoneField(g, T)
    S = ZoneField(g, T)
    ρ = ZoneField(g, T)
    p = ZoneField(g, T)
    pHY = ZoneField(g, T)
    pHY′ = ZoneField(g, T)
    pNHS = ZoneField(g, T)
    Fields(u, v, w, θ, S, ρ, p, pHY, pHY′, pNHS)
end

struct SourceTermFields{T <: AbstractFloat} <: FieldCollection
    Gu::FaceField{T}
    Gv::FaceField{T}
    Gw::FaceField{T}
    Gθ::ZoneField{T}
    GS::ZoneField{T}
end

function SourceTermFields(g::RegularCartesianGrid, T=Float64)
    Gu = FaceField(g, T)
    Gv = FaceField(g, T)
    Gw = FaceField(g, T)
    Gθ = ZoneField(g, T)
    GS = ZoneField(g, T)
    SourceTermFields{T}(Gu, Gv, Gw, Gθ, GS)
end

struct ForcingFields{T <: AbstractFloat} <: FieldCollection
    Fu::FaceField{T}
    Fv::FaceField{T}
    Fw::FaceField{T}
    Fθ::ZoneField{T}
    FS::ZoneField{T}
end

function ForcingFields(g::RegularCartesianGrid, T=Float64)
    Fu = FaceField(g, T)
    Fv = FaceField(g, T)
    Fw = FaceField(g, T)
    Fθ = ZoneField(g, T)
    FS = ZoneField(g, T)
end
