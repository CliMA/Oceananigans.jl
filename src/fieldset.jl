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
    fC4::CellField
    fFX::FaceFieldX
    fFY::FaceFieldY
    fFZ::FaceFieldZ
    fCC1::CellField
    fCC2::CellField
end

function VelocityFields(g)
    u = FaceFieldX(g)
    v = FaceFieldY(g)
    w = FaceFieldZ(g)
    VelocityFields(u, v, w)
end

function TracerFields(g)
    θ = CellField(g)
    S = CellField(g)
    ρ = CellField(g)
    TracerFields(θ, S, ρ)
end

function PressureFields(g)
    pHY = CellField(g)
    pHY′ = CellField(g)
    pNHS = CellField(g)
    PressureFields(pHY, pHY′, pNHS)
end

function SourceTerms(g)
    Gu = FaceFieldX(g)
    Gv = FaceFieldY(g)
    Gw = FaceFieldZ(g)
    GT = CellField(g)
    GS = CellField(g)
    SourceTerms(Gu, Gv, Gw, GT, GS)
end

function ForcingFields(g)
    Fu = FaceFieldX(g)
    Fv = FaceFieldY(g)
    Fw = FaceFieldZ(g)
    FT = CellField(g)
    FS = CellField(g)
    ForcingFields(Fu, Fv, Fw, FT, FS)
end

function TemporaryFields(g)
    fC1 = CellField(g)
    fC2 = CellField(g)
    fC3 = CellField(g)
    fC4 = CellField(g)
    fFX = FaceFieldX(g)
    fFY = FaceFieldY(g)
    fFZ = FaceFieldZ(g)
    fCC1 = CellField(g, Complex{eltype(g)})
    fCC2 = CellField(g, Complex{eltype(g)})
    TemporaryFields(fC1, fC2, fC3, fC4, fFX, fFY, fFZ, fCC1, fCC2)
end
