using Oceananigans

####
#### Definition of a compressible model
####

mutable struct CompressibleModel{A, FT, G, BS, M, D, E, MR, F, R, P} <: AbstractModel
             architecture :: A
                     grid :: G
         surface_pressure :: FT
               base_state :: BS
                  momenta :: M
                densities :: D
                  tracers :: T
            slow_forcings :: F
            fast_forcings :: R
    acoustic_time_stepper :: P
end

####
#### Constructor for compressible models
####

function CompressibleModel(;
                     grid,
             architecture = CPU(),
         surface_pressure = 100000,
               base_state = nothing,
                  momenta = MomentumFields(arch, grid),
                densities = DensityFields(arch, grid),
                  tracers = (:S, :Qv, :Ql, :Qi),
            slow_forcings = ForcingFields(arch, grid, tracernames(tracers)),
            fast_forcings = ForcingFields(arch, grid, tracernames(tracers)),
    acoustic_time_stepper = nothing
    )

    tracers = TracerFields(architecture, grid, tracers)

    return CompressibleModel(architecture, grid, surface_pressure, base_state,
                             momenta, densities, tracers, slow_forcings, fast_forcings,
                             acoustic_time_stepper)
end

####
#### Utilities for constructing compressible models
####

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

tracernames(::Nothing) = ()
tracernames(name::Symbol) = tuple(name)
tracernames(::NamedTuple{names}) where names = tracernames(names)

function TracerFields(arch, grid, tracernames)
    tracerfields = Tuple(CellField(arch, grid) for c in tracernames)
    return NamedTuple{tracernames}(tracerfields)
end

function ForcingFields(arch, grid, tracernames)
    U = FaceFieldX(arch, grid)
    V = FaceFieldY(arch, grid)
    W = FaceFieldZ(arch, grid)
    momenta = (U=U, V=V, W=W)

    tracers = TracerFields(arch, grid, tracernames)

    return merge(momenta, tracers)
end

