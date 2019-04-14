struct VelocityFields <: FieldSet
    u::FaceFieldX
    v::FaceFieldY
    w::FaceFieldZ
end

struct TracerFields <: FieldSet
    T::CellField
    S::CellField
end

struct PressureFields <: FieldSet
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

function VelocityFields(arch::Architecture, grid::Grid)
    u = FaceFieldX(arch, grid)
    v = FaceFieldY(arch, grid)
    w = FaceFieldZ(arch, grid)
    VelocityFields(u, v, w)
end

function TracerFields(arch::Architecture, grid::Grid)
    θ = CellField(arch, grid)  # Temperature θ to avoid conflict with type T.
    S = CellField(arch, grid)
    TracerFields(θ, S)
end

function PressureFields(arch::Architecture, grid::Grid)
    pHY′ = CellField(arch, grid)
    pNHS = CellField(arch, grid)
    PressureFields(pHY′, pNHS)
end

function SourceTerms(arch::Architecture, grid::Grid)
    Gu = FaceFieldX(arch, grid)
    Gv = FaceFieldY(arch, grid)
    Gw = FaceFieldZ(arch, grid)
    GT = CellField(arch, grid)
    GS = CellField(arch, grid)
    SourceTerms(Gu, Gv, Gw, GT, GS)
end
