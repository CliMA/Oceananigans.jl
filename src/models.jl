using Oceananigans

mutable struct CompressibleModel{A, FT, G, M, P, T, F, R} <: AbstractModel
    architecture     :: A
    grid             :: G
    surface_pressure :: FT
    momenta          :: M
    densities        :: P
    tracers          :: T
    slow_forcings    :: F
    fast_forcings    :: R
end

function CompressibleModel(; architecture=CPU(), grid, surface_pressure = 100000)
    Ũ = MomentumFields(arch, grid) 
    ρ = DensityFields(arch, grid)
    C = TracerFields(arch, grid)
    F = SlowForcingFields(arch, grid)
    R = FastForcingFields(arch, grid)
    return CompressibleModel(architecture, grid, surface_pressure, Ũ, ρ, C, F, R)
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

function TracerFields(arch, grid)
    S = CellField(arch, grid)
    Qv = CellField(arch, grid)
    Ql = CellField(arch, grid)
    Qi = CellField(arch, grid)
    return (S=S, Qv=Qv, Ql=Ql, Qi=Qi)
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

function 
