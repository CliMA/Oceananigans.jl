using Oceananigans

mutable struct CompressibleModel{A, FT, G, BS, M, D, E, MR, F, R, P} <: AbstractModel
             architecture :: A
                     grid :: G
         surface_pressure :: FT
               base_state :: BS
                  momenta :: M
                densities :: D
                  entropy :: E
            mixing_ratios :: MR
            slow_forcings :: F
            fast_forcings :: R
    acoustic_time_stepper :: P
end

function CompressibleModel(;
                     grid,
             architecture = CPU(),
         surface_pressure = 100000,
               base_state = nothing,
                  momenta = MomentumFields(arch, grid),
                densities = DensityFields(arch, grid),
                  entropy = CellField(arch, grid),
            mixing_ratios = MixingRatioFields(arch, grid),
            slow_forcings = SlowForcingFields(arch, grid),
            fast_forcings = FastForcingFields(arch, grid),
    acoustic_time_stepper = nothing
    )

    return CompressibleModel(architecture, grid, surface_pressure, base_state,
                             momenta, densities, entropy, mixing_ratios, slow_forcings,
                             fast_forcings, acoustic_perturbation, acoustic_time_stepper)
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

function AcousticPerturbationFields(arch, grid)
    U″ = FaceFieldX(arch, grid)
    V″ = FaceFieldY(arch, grid)
    W″ = FaceFieldZ(arch, grid)
    S″ = CellField(arch, grid)
    ρ″ = CellField(arch, grid)
    return (U=U″, V=V″, W=W″, S=S″, ρ=ρ″)
end
