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

struct StepperTemporaryFields <: FieldSet
    fC1::CellField
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

function StepperTemporaryFields(metadata::ModelMetadata, grid::Grid)
    fC1 = CellField(metadata, grid, metadata.float_type)
    fCC1 = CellField(metadata, grid, Complex{Float64})  # We might really need Float64 for the Poisson solver.
    fCC2 = CellField(metadata, grid, Complex{Float64})
    StepperTemporaryFields(fC1, fCC1, fCC2)
end
