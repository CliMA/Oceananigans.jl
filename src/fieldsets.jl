struct VelocityFields{U<:FaceFieldX, V<:FaceFieldY, W<:FaceFieldZ} <: FieldSet
    u::U
    v::V
    w::W
end

struct TracerFields{CFT<:CellField, CFS<:CellField} <: FieldSet
    T::CFT
    S::CFS
end

struct PressureFields{CF1<:CellField, CF2<:CellField} <: FieldSet
    pHY′::CF1
    pNHS::CF2
end

struct SourceTerms{U<:FaceFieldX, V<:FaceFieldY, W<:FaceFieldZ, T<:CellField, S<:CellField} <: FieldSet
    Gu::U
    Gv::V
    Gw::W
    GT::T
    GS::S
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
