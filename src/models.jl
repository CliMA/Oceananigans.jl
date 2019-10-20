using Oceananigans

mutable struct CompressibleModel{A, FT, G, M, E, ρ, T, MR, F, R, P} <: AbstractModel
    architecture     :: A
    grid             :: G
    surface_pressure :: FT
    momenta          :: M
    entropy          :: E
    densities        :: ρ
    mixing_ratios    :: MR
    slow_forcings    :: F
    fast_forcings    :: R
    perturbations    :: P
end

function CompressibleModel(; architecture=CPU(), grid, surface_pressure = 100000)
    Ũ = MomentumFields(arch, grid)
    S = CellField(arch, grid)
    ρ = DensityFields(arch, grid)
    Q = MixingRatioFields(arch, grid)
    F = SlowForcingFields(arch, grid)
    R = FastForcingFields(arch, grid)
    P = PerturbationFields(arch, grid)
    return CompressibleModel(architecture, grid, surface_pressure, Ũ, S, ρ, Q, F, R, P)
end

function MomentumFields(arch, grid)
    U = FaceFieldX(arch, grid)
    V = FaceFieldY(arch, grid)
    W = FaceFieldZ(arch, grid)
    return (U=U, V=V, W=W)
end

function DensityFields(arch, grid)
    ρd = CellField(arch, grid)
    ρm = CellField(arch, grid)
    return (d=ρd, m=ρm)
end

function MixingRatioFields(arch, grid)
    Qv = CellField(arch, grid)
    Ql = CellField(arch, grid)
    Qi = CellField(arch, grid)
    return (Qv=Qv, Ql=Ql, Qi=Qi)
end

function SlowForcingFields(arch, grid)
    U = FaceFieldX(arch, grid)
    V = FaceFieldY(arch, grid)
    W = FaceFieldZ(arch, grid)
    S = CellField(arch, grid)
    Qv = CellField(arch, grid)
    Ql = CellField(arch, grid)
    Qi = CellField(arch, grid)
    return (U=U, V=V, W=W, S=S, Qv=Qv, Ql=Ql, Qi=Qi)
end

function FastForcingFields(arch, grid)
    U = FaceFieldX(arch, grid)
    V = FaceFieldY(arch, grid)
    W = FaceFieldZ(arch, grid)
    S = CellField(arch, grid)
    ρd = CellField(arch, grid)
    Qv = CellField(arch, grid)
    Ql = CellField(arch, grid)
    Qi = CellField(arch, grid)
    return (U=U, V=V, W=W, ρd=ρd, S=S, Qv=Qv, Ql=Ql, Qi=Qi)
end

function PerturbationFields(arch, grid)
    U″ = FaceFieldX(arch, grid)
    V″ = FaceFieldY(arch, grid)
    W″ = FaceFieldZ(arch, grid)
    S″ = CellField(arch, grid)
    ρ″ = CellField(arch, grid)
    return (U=U″, V=V″, W=W″, S=S″, ρ=ρ″)
end
