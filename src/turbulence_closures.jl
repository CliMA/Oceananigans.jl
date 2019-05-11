abstract type TurbulenceClosure end

struct NoClosure <: TurbulenceClosure
    ν_turb :: Nothing
end

NoClosure() = NoClosure(nothing)

struct ConstantAnisotropicDiffusion{T} <: TurbulenceClosure
    𝜈h :: T
    𝜈v :: T
    κh :: T
    κv :: T
end

struct ConstantSmagorinsky{T, TV, TS} <: TurbulenceClosure
     ν_turb :: TV
     strain :: TS
    prandtl :: T
      coeff :: T
end

function ConstantSmagorinsky(arch, grid; prandtl=1.0, coeff=0.13)
    ν_turb = TurbulentViscosity(arch, grid)
    ConstantSmagorinsky(ν_turb, prandtl, coeff)
end

const C = Cell
const F = Interface

#=
Note:

u₁ :: u :: FCC
u₂ :: v :: CFC
u₃ :: w :: CCF

Sᵢⱼ = ∂ⱼ uᵢ

therefore

S₁₁ :: S₂₂ :: S₃₃ :: CCC
S₁₂ :: S₂₁ :: FFC
S₁₃ :: S₃₁ :: FCF
S₂₃ :: S₃₂ :: CFF

finally,

S66 = S₁₁² + S₂₂² + S₃₃².
=#

struct TurbulentViscosity{A, G}
    𝜈CCC :: CellField{A, G}
    𝜈FFC :: GeneralField{F, F, C, A, G}
    𝜈FCF :: GeneralField{F, C, F, A, G}
end

function TurbulentViscosity(arch, grid)
    𝜈CCC = CellField(arch, grid)
    𝜈FFC = GeneralField(F, F, C, arch, grid)
    𝜈FCF = GeneralField(F, C, F, arch, grid)
    TurbulentViscosity(𝜈CCC, 𝜈FFC, 𝜈FCF)
end

struct Strain{A, G}
    S12 :: GeneralField{F, F, C, A, G}
    S13 :: GeneralField{F, C, F, A, G}
    S23 :: GeneralField{C, F, F, A, G}
    S66 :: CellField{A, G}
end

function Strain(arch, grid)
    S12 = GeneralField(F, F, C, arch, grid)
    S13 = GeneralField(F, C, F, arch, grid)
    S23 = GeneralField(C, F, F, arch, grid)
    S66 = CellField(arch, grid)
    return Strain(S12, S13, S23, S66)
end
