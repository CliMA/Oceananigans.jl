using Oceananigans

using Oceananigans.Models: AbstractModel, Clock
using Oceananigans.Forcing: zeroforcing

#####
##### Definition of a compressible model
#####

mutable struct CompressibleModel{A, FT, G, M, D, T, TS, TD, C, TC, DF, MP, F, P, SF, RHS, IV, ATS} <: AbstractModel
             architecture :: A
                     grid :: G
                    clock :: Clock{FT}
                  momenta :: M
                densities :: D
   thermodynamic_variable :: T
             microphysics :: MP
                  tracers :: TS
            total_density :: TD
                 coriolis :: C
                  closure :: TC
            diffusivities :: DF
                  forcing :: F
               parameters :: P
                  gravity :: FT
            slow_forcings :: SF
         right_hand_sides :: RHS
        intermediate_vars :: IV
    acoustic_time_stepper :: ATS
end

#####
##### Constructor for compressible models
#####

const pₛ_Earth = 100000
const  g_Earth = 9.80665

function CompressibleModel(;
                     grid,
             architecture = CPU(),
               float_type = Float64,
                    clock = Clock{float_type}(0, 0),
                  momenta = MomentumFields(architecture, grid),
                densities = DryEarth(float_type),
   thermodynamic_variable = PrognosticEntropy(),
             microphysics = nothing,
            extra_tracers = nothing,
              tracernames = collect_tracers(densities, thermodynamic_variable, microphysics, extra_tracers),
                 coriolis = nothing,
                  closure = ConstantIsotropicDiffusivity(float_type, ν=0.5, κ=0.5),
            diffusivities = TurbulentDiffusivities(architecture, grid, tracernames, closure),
                  forcing = ModelForcing(),
               parameters = nothing,
                  gravity = g_Earth,
            slow_forcings = ForcingFields(architecture, grid, tracernames),
         right_hand_sides = RightHandSideFields(architecture, grid, tracernames),
        intermediate_vars = RightHandSideFields(architecture, grid, tracernames),
    acoustic_time_stepper = nothing
   )

    gravity = float_type(gravity)
    tracers = TracerFields(architecture, grid, tracernames)
    forcing = ModelForcing(tracernames, forcing)
    closure = with_tracers(tracernames, closure)
    total_density = CellField(architecture, grid)

    return CompressibleModel(architecture, grid, clock, momenta, densities, thermodynamic_variable,
                             microphysics, tracers, total_density, coriolis, closure, diffusivities,
                             forcing, parameters, gravity, slow_forcings,
                             right_hand_sides, intermediate_vars, acoustic_time_stepper)
end

#####
##### Utilities for constructing compressible models
#####

function collect_tracers(args...)
    list = []
    for arg in args
        if arg !== nothing
            for name in keys(arg)
                push!(list, name)
            end
        end
    end
    return Tuple(list)
end

function DryEarth(FT = Float64)
    return (ρ = EarthN₂O₂(FT),)
end

function PrognosticS()
    return (ρs = Entropy(),)
end

function MomentumFields(arch, grid)
    ρu = FaceFieldX(arch, grid)
    ρv = FaceFieldY(arch, grid)
    ρw = FaceFieldZ(arch, grid)
    return (ρu=ρu, ρv=ρv, ρw=ρw)
end

function TracerFields(arch, grid, tracernames)
    tracerfields = Tuple(CellField(arch, grid) for c in tracernames)
    return NamedTuple{tracernames}(tracerfields)
end

function ForcingFields(arch, grid, tracernames)
    ρu = FaceFieldX(arch, grid)
    ρv = FaceFieldY(arch, grid)
    ρw = FaceFieldZ(arch, grid)
    momenta = (ρu=ρu, ρv=ρv, ρw=ρw)
    tracers = TracerFields(arch, grid, tracernames)
    return merge(momenta, tracers)
end

function RightHandSideFields(arch, grid, tracernames)
    ρu = FaceFieldX(arch, grid)
    ρv = FaceFieldY(arch, grid)
    ρw = FaceFieldZ(arch, grid)
    momenta = (ρu=ρu, ρv=ρv, ρw=ρw)
    tracers = TracerFields(arch, grid, tracernames)
    return merge(momenta, tracers)
end
