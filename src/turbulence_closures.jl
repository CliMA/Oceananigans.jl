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

struct TurbulentViscosity{A, G}
    𝜈00 :: CellField{A, G}
    𝜈12 :: CellField{A, G}
    𝜈13 :: CellField{A, G}
    𝜈23 :: EdgeField{A, G}
end

function TurbulentViscosity(arch, grid)
    𝜈00 = CellField(arch, grid)
    𝜈12 = EdgeField(arch, grid)
    𝜈13 = EdgeField(arch, grid)
    𝜈23 = EdgeField(arch, grid)
    TurbulentViscosity(𝜈00 , 𝜈12 , 𝜈13 , 𝜈23)
end

struct Strain{A}
    S12 :: A
    S13 :: A
    S23 :: A
end

import Base: zeros

function zeros(T, ::GPU, g)
    a = CuArray{T}(undef, g.Nx, g.Ny, g.Nz)
    a .= 0
    return a
end

zeros(T, ::CPU, g) where T = zeros(T, size(g))

# Default to type of Grid
zeros(arch::Architecture, g::Grid{T}) where T = zeros(T, arch, g)

Strain(arch, grid) = Strain((zeros((arch, grid) for i=1:3)...)
