using Oceananigans

using Oceananigans: AbstractModel

####
#### Definition of a compressible model
####

mutable struct CompressibleModel{A, FT, G, B, C, BS, M, D, T, SF, FF, IV, P} <: AbstractModel
             architecture :: A
                     grid :: G
                    clock :: Clock{FT}
                 buoyancy :: B
                 coriolis :: C
         surface_pressure :: FT
               base_state :: BS
                  momenta :: M
                densities :: D
                  tracers :: T
            slow_forcings :: SF
            fast_forcings :: FF
        intermediate_vars :: IV
    acoustic_time_stepper :: P
end

####
#### Constructor for compressible models
####

function CompressibleModel(;
                     grid,
             architecture = CPU(),
               float_type = Float64,
                    clock = Clock{float_type}(0, 0),
                 buoyancy = DryIdealGas(float_type),
                 coriolis = nothing,
         surface_pressure = 100000,
               base_state = nothing,
                  momenta = MomentumFields(architecture, grid),
                densities = DensityFields(architecture, grid),
                  tracers = (:S, :Qv, :Ql, :Qi),
            slow_forcings = ForcingFields(architecture, grid, tracernames(tracers)),
            fast_forcings = ForcingFields(architecture, grid, tracernames(tracers)),
        intermediate_vars = IntermediateFields(architecture, grid, tracernames(tracers)),
    acoustic_time_stepper = nothing
    )

    surface_pressure = float_type(surface_pressure)
    tracers = TracerFields(architecture, grid, tracers)

    return CompressibleModel(architecture, grid, clock, buoyancy, coriolis, surface_pressure,
                             base_state, momenta, densities, tracers, slow_forcings,
                             fast_forcings, acoustic_time_stepper)
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
tracernames(names::NTuple{N, Symbol}) where N = names
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

function IntermediateFields(architecture, grid, tracernames(tracers))
    U = FaceFieldX(arch, grid)
    V = FaceFieldY(arch, grid)
    W = FaceFieldZ(arch, grid)
    ρ =  CellField(arch, grid)
    momenta_and_density = (U=U, V=V, W=W, ρ=ρ)

    tracers = TracerFields(arch, grid, tracernames)

    return merge(momenta_and_density, tracers)
end
