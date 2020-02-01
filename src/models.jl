using Oceananigans

using Oceananigans.Models: AbstractModel, Clock
using Oceananigans.Forcing: zeroforcing

#####
##### Definition of a compressible model
#####

mutable struct CompressibleModel{A, FT, G, M, D, T, TS, MX, B, C, TC, DF, F, P, SF, RHS, IV, ATS} <: AbstractModel
             architecture :: A
                     grid :: G
                    clock :: Clock{FT}
                  momenta :: M
                  density :: D
   prognostic_temperature :: T
                  tracers :: TS
            mixing_ratios :: MX
                 buoyancy :: B
                 coriolis :: C
                  closure :: TC
            diffusivities :: DF
                  forcing :: F
               parameters :: P
       reference_pressure :: FT
            slow_forcings :: SF
         right_hand_sides :: RHS
        intermediate_vars :: IV
    acoustic_time_stepper :: ATS
end

#####
##### Constructor for compressible models
#####

function CompressibleModel(;
                     grid,
             architecture = CPU(),
               float_type = Float64,
                    clock = Clock{float_type}(0, 0),
                  momenta = MomentumFields(architecture, grid),
                  density = CellField(architecture, grid),
   prognostic_temperature = ModifiedPotentialTemperature(),
                  tracers = (:Θᵐ,),
            mixing_ratios = (:Qv, :Ql, :Qi),
                 buoyancy = IdealGas(float_type),
                 coriolis = nothing,
                  closure = ConstantIsotropicDiffusivity(float_type, ν=0.5, κ=0.5),
            diffusivities = TurbulentDiffusivities(architecture, grid, tracernames(tracers), closure),
                  forcing = ModelForcing(),
               parameters = nothing,
       reference_pressure = 100000,
            slow_forcings = ForcingFields(architecture, grid, tracernames(tracers)),
         right_hand_sides = RightHandSideFields(architecture, grid, tracernames(tracers)),
        intermediate_vars = RightHandSideFields(architecture, grid, tracernames(tracers)),
    acoustic_time_stepper = nothing
   )

    validate_prognostic_temperature(prognostic_temperature, tracers)

    reference_pressure = float_type(reference_pressure)
    tracers = TracerFields(architecture, grid, tracers)
    mixing_ratios = MixingRatioFields(arch, grid, mixing_ratios)

    forcing = ModelForcing(tracernames(tracers), forcing)
    closure = with_tracers(tracernames(tracers), closure)

    return CompressibleModel(architecture, grid, clock, momenta, density, prognostic_temperature,
                             tracers, mixing_ratios, buoyancy, coriolis, closure, diffusivities,
                             forcing, parameters, reference_pressure, slow_forcings, right_hand_sides,
                             intermediate_vars, acoustic_time_stepper)
end

#####
##### Utilities for constructing compressible models
#####

function MomentumFields(arch, grid)
    ρu = FaceFieldX(arch, grid)
    ρv = FaceFieldY(arch, grid)
    ρw = FaceFieldZ(arch, grid)
    return (ρu=ρu, ρv=ρv, ρw=ρw)
end

tracernames(::Nothing) = ()
tracernames(name::Symbol) = tuple(name)
tracernames(names::NTuple{N, Symbol}) where N = names
tracernames(::NamedTuple{names}) where names = tracernames(names)

function TracerFields(arch, grid, tracernames)
    tracerfields = Tuple(CellField(arch, grid) for c in tracernames)
    return NamedTuple{tracernames}(tracerfields)
end

function MixingRatioFields(arch, grid, mixing_ratio_names)
    mixing_ratio_fields = Tuple(CellField(arch, grid) for c in mixing_ratio_names)
    return NamedTuple{mixing_ratio_names}(mixing_ratio_fields)
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
    ρ =  CellField(arch, grid)
    momenta_and_density = (ρu=ρu, ρv=ρv, ρw=ρw, ρ=ρ)

    tracers = TracerFields(arch, grid, tracernames)

    return merge(momenta_and_density, tracers)
end
