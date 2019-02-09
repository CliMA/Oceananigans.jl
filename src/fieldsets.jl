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

struct OperatorTemporaryFields <: FieldSet
    fC1::CellField
    fC2::CellField
    fC3::CellField
    fC4::CellField
    fFX::FaceFieldX
    fFX2::FaceFieldX
    fFY::FaceFieldY
    fFY2::FaceFieldX
    fFZ::FaceFieldZ
    fFZ2::FaceFieldX
    fE1::EdgeField
    fE2::EdgeField
end

struct StepperTemporaryFields <: FieldSet
    fC1::CellField
    fC2::CellField
    fC3::CellField
    fC4::CellField
    fFX::FaceFieldX
    fFX2::FaceFieldX
    fFY::FaceFieldY
    fFY2::FaceFieldX
    fFZ::FaceFieldZ
    fFZ2::FaceFieldX
    fCC1::CellField
    fCC2::CellField
end

function VelocityFields(metadata::ModelMetadata, grid::Grid)
    u = FaceFieldX(metadata, grid)
    v = FaceFieldY(metadata, grid)
    w = FaceFieldZ(metadata, grid)
    VelocityFields(u, v, w)
end

function TracerFields(metadata::ModelMetadata, grid::Grid)
    θ = CellField(metadata, grid)
    S = CellField(metadata, grid)
    ρ = CellField(metadata, grid)
    TracerFields(θ, S, ρ)
end

function PressureFields(metadata::ModelMetadata, grid::Grid)
    pHY = CellField(metadata, grid)
    pHY′ = CellField(metadata, grid)
    pNHS = CellField(metadata, grid)
    PressureFields(pHY, pHY′, pNHS)
end

function SourceTerms(metadata::ModelMetadata, grid::Grid)
    Gu = FaceFieldX(metadata, grid)
    Gv = FaceFieldY(metadata, grid)
    Gw = FaceFieldZ(metadata, grid)
    GT = CellField(metadata, grid)
    GS = CellField(metadata, grid)
    SourceTerms(Gu, Gv, Gw, GT, GS)
end

function ForcingFields(metadata::ModelMetadata, grid::Grid)
    Fu = FaceFieldX(metadata, grid)
    Fv = FaceFieldY(metadata, grid)
    Fw = FaceFieldZ(metadata, grid)
    FT = CellField(metadata, grid)
    FS = CellField(metadata, grid)
    ForcingFields(Fu, Fv, Fw, FT, FS)
end

function OperatorTemporaryFields(metadata::ModelMetadata, grid::Grid)
    fC1 = CellField(metadata, grid)
    fC2 = CellField(metadata, grid)
    fC3 = CellField(metadata, grid)
    fC4 = CellField(metadata, grid)
    fFX = FaceFieldX(metadata, grid)
    fFX2 = FaceFieldX(metadata, grid)
    fFY = FaceFieldY(metadata, grid)
    fFY2 = FaceFieldX(metadata, grid)
    fFZ = FaceFieldZ(metadata, grid)
    fFZ2 = FaceFieldX(metadata, grid)
    fE1 = EdgeField(metadata, grid)
    fE2 = EdgeField(metadata, grid)
    OperatorTemporaryFields(fC1, fC2, fC3, fC4, fFX, fFX2, fFY, fFY2, fFZ, fFZ2, fE1, fE2)
end

function StepperTemporaryFields(metadata::ModelMetadata, grid::Grid)
    fC1 = CellField(metadata, grid)
    fC2 = CellField(metadata, grid)
    fC3 = CellField(metadata, grid)
    fC4 = CellField(metadata, grid)
    fFX = FaceFieldX(metadata, grid)
    fFX2 = FaceFieldX(metadata, grid)
    fFY = FaceFieldY(metadata, grid)
    fFY2 = FaceFieldX(metadata, grid)
    fFZ = FaceFieldZ(metadata, grid)
    fFZ2 = FaceFieldX(metadata, grid)
    fCC1 = CellField(metadata, grid, Complex{metadata.float_type})
    fCC2 = CellField(metadata, grid, Complex{metadata.float_type})
    StepperTemporaryFields(fC1, fC2, fC3, fC4, fFX, fFX2, fFY, fFY2, fFZ, fFZ2, fCC1, fCC2)
end
