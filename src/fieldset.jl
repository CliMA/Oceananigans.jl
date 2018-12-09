struct VelocityFields <: FieldSet
    u::FaceFieldX
    v::FaceFieldY
    w::FaceFieldZ
end

struct TracerFields <: FieldSet
    T::CellField
    S::CellField
    ρ::CellField
end

struct PressureFields <: FieldSet
    pHY::CellField
    pHY′::CellField
    pNHS::CellField
end

struct SourceTerms <: FieldSet
    Gu::FaceFieldX
    Gv::FaceFieldY
    Gw::FaceFieldZ
    GT::CellField
    GS::CellField
end

struct ForcingFields <: FieldSet
    Fu::FaceFieldX
    Fv::FaceFieldY
    Fw::FaceFieldZ
    FT::CellField
    FS::CellField
end

struct TemporaryFields <: FieldSet
    fC1::CellField
    fC2::CellField
    fC3::CellField
    ffX::FaceFieldX
    ffY::FaceFieldY
    ffZ::FaceFieldZ
end

function VelocityFields(g, T=Float64)
    u = FaceFieldX(g, T)
    v = FaceFieldY(g, T)
    w = FaceFieldZ(g, T)
    VelocityFields(u, v, w)
end

function TracerFields(g, T=Float64)
    T = CellField(g ,T)
    S = CellField(g ,T)
    ρ = CellField(g ,T)
    TracerField(T, S, ρ)
end

function PressureFields(g, T=Float64)
    pHY = CellField(g ,T)
    pHY′ = CellField(g ,T)
    pNHS = CellField(g ,T)
    PressureFields(pHY, pHY′, pNHS)
end

function SourceTerms(g, T=Float64)
    Gu = FaceFieldX(g, T)
    Gv = FaceFieldY(g, T)
    Gw = FaceFieldZ(g, T)
    GT = CellField(T)
    GS = CellField(S)
    SourceTerms(Gu, Gv, Gw, GT, GS)
end

function ForcingFields(g, T=Float64)
    Fu = FaceFieldX(g, T)
    Fv = FaceFieldY(g, T)
    Fw = FaceFieldZ(g, T)
    FT = CellField(T)
    FS = CellField(S)
    ForcingFields(Fu, Fv, Fw, FT, FS)
end

function TemporaryFields(g, T=Float64)
    fC1 = CellField(g, T)
    fC2 = CellField(g, T)
    fC3 = CellField(g, T)
    fFX = FaceFieldX(g, T)
    fFY = FaceFieldY(g, T)
    fFZ = FaceFieldZ(g, T)
    TemporaryFields(fC1, fC2, fC3, fFX, fFY, fFZ)
end
