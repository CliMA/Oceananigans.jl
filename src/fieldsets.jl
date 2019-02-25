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
    u = FaceFieldX(metadata, grid, metadata.float_type)
    v = FaceFieldY(metadata, grid, metadata.float_type)
    w = FaceFieldZ(metadata, grid, metadata.float_type)
    VelocityFields(u, v, w)
end

function TracerFields(metadata::ModelMetadata, grid::Grid)
    θ = CellField(metadata, grid, metadata.float_type)
    S = CellField(metadata, grid, metadata.float_type)
    ρ = CellField(metadata, grid, metadata.float_type)
    TracerFields(θ, S, ρ)
end

function PressureFields(metadata::ModelMetadata, grid::Grid)
    pHY = CellField(metadata, grid, metadata.float_type)
    pHY′ = CellField(metadata, grid, metadata.float_type)
    pNHS = CellField(metadata, grid, metadata.float_type)
    PressureFields(pHY, pHY′, pNHS)
end

function SourceTerms(metadata::ModelMetadata, grid::Grid)
    Gu = FaceFieldX(metadata, grid, metadata.float_type)
    Gv = FaceFieldY(metadata, grid, metadata.float_type)
    Gw = FaceFieldZ(metadata, grid, metadata.float_type)
    GT = CellField(metadata, grid, metadata.float_type)
    GS = CellField(metadata, grid, metadata.float_type)
    SourceTerms(Gu, Gv, Gw, GT, GS)
end

function ForcingFields(metadata::ModelMetadata, grid::Grid)
    Fu = FaceFieldX(metadata, grid, metadata.float_type)
    Fv = FaceFieldY(metadata, grid, metadata.float_type)
    Fw = FaceFieldZ(metadata, grid, metadata.float_type)
    FT = CellField(metadata, grid, metadata.float_type)
    FS = CellField(metadata, grid, metadata.float_type)
    ForcingFields(Fu, Fv, Fw, FT, FS)
end

function OperatorTemporaryFields(metadata::ModelMetadata, grid::Grid)
    fC1 = CellField(metadata, grid, metadata.float_type)
    fC2 = CellField(metadata, grid, metadata.float_type)
    fC3 = CellField(metadata, grid, metadata.float_type)
    fC4 = CellField(metadata, grid, metadata.float_type)
    fFX = FaceFieldX(metadata, grid, metadata.float_type)
    fFX2 = FaceFieldX(metadata, grid, metadata.float_type)
    fFY = FaceFieldY(metadata, grid, metadata.float_type)
    fFY2 = FaceFieldX(metadata, grid, metadata.float_type)
    fFZ = FaceFieldZ(metadata, grid, metadata.float_type)
    fFZ2 = FaceFieldX(metadata, grid, metadata.float_type)
    fE1 = EdgeField(metadata, grid, metadata.float_type)
    fE2 = EdgeField(metadata, grid, metadata.float_type)
    OperatorTemporaryFields(fC1, fC2, fC3, fC4, fFX, fFX2, fFY, fFY2, fFZ, fFZ2, fE1, fE2)
end

function StepperTemporaryFields(metadata::ModelMetadata, grid::Grid)
    fC1 = CellField(metadata, grid, metadata.float_type)
    fC2 = CellField(metadata, grid, metadata.float_type)
    fC3 = CellField(metadata, grid, metadata.float_type)
    fC4 = CellField(metadata, grid, metadata.float_type)
    fFX = FaceFieldX(metadata, grid, metadata.float_type)
    fFX2 = FaceFieldX(metadata, grid, metadata.float_type)
    fFY = FaceFieldY(metadata, grid, metadata.float_type)
    fFY2 = FaceFieldX(metadata, grid, metadata.float_type)
    fFZ = FaceFieldZ(metadata, grid, metadata.float_type)
    fFZ2 = FaceFieldX(metadata, grid, metadata.float_type)
    fCC1 = CellField(metadata, grid, Complex{Float64})  # We might really need Float64 for the Poisson solver.
    fCC2 = CellField(metadata, grid, Complex{Float64})
    StepperTemporaryFields(fC1, fC2, fC3, fC4, fFX, fFX2, fFY, fFY2, fFZ, fFZ2, fCC1, fCC2)
end
