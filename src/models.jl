import Base: getproperty

using Oceananigans
using Oceananigans.Models: AbstractModel, Clock
using Oceananigans.Forcing: zeroforcing

#####
##### Definition of a compressible model
#####

mutable struct CompressibleModel{A, FT, G, M, GS, TV, TF, TS, D, TD, C, TC, DF, MP, F, SF, RHS, IV, ATS} <: AbstractModel
             architecture :: A
                     grid :: G
                    clock :: Clock{FT}
                  momenta :: M
                    gases :: GS
   thermodynamic_variable :: TV
             microphysics :: MP
                  tracers :: TS
           density_fields :: D
     thermodynamic_fields :: TF
            total_density :: TD
                 coriolis :: C
                  closure :: TC
            diffusivities :: DF
                  forcing :: F
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
                    gases = DryEarth(float_type),
   thermodynamic_variable = Energy(),
             microphysics = nothing,
            extra_tracers = nothing,
              tracernames = collect_tracers(thermodynamic_variable, gases, microphysics, extra_tracers),
                 coriolis = nothing,
                  closure = ConstantIsotropicDiffusivity(float_type, ν=0.5, κ=0.5),
            diffusivities = TurbulentDiffusivities(architecture, grid, tracernames, closure),
                  forcing = ModelForcing(),
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
    density_fields = create_density_pointers(gases, tracers,
        forcing, slow_forcings, right_hand_sides, intermediate_vars)
    thermodynamic_fields = create_thermodynamic_pointers(
        thermodynamic_variable, tracers, forcing, slow_forcings,
        right_hand_sides, intermediate_vars)
    total_density = CellField(architecture, grid)

    return CompressibleModel(architecture, grid, clock, momenta, gases, thermodynamic_variable,
                             microphysics, tracers, density_fields, thermodynamic_fields,
                             total_density, coriolis, closure, diffusivities,
                             forcing, gravity, slow_forcings,
                             right_hand_sides, intermediate_vars, acoustic_time_stepper)
end

#####
##### Utilities for constructing compressible models
#####

function collect_tracers(thermodynamic_variable, args...)
    list = []
    push!(list, generate_key(thermodynamic_variable))
    for arg in args
        if arg !== nothing
            for name in keys(arg)
                push!(list, name)
            end
        end
    end
    return Tuple(list)
end

function create_density_pointers(gases, tracers, forcings, slow_forcings,
    right_hand_sides, intermediate_vars)
    names = Tuple(key for key in keys(gases))
    values = Tuple((gas = getproperty(gases, key),
                    values = getproperty(tracers, key),
                    forcing = getproperty(forcings, key),
                    F = getproperty(slow_forcings, key),
                    R = getproperty(right_hand_sides, key),
                    IV = getproperty(intermediate_vars, key))
                    for key in keys(gases))
    return NamedTuple{names}(values)
end

function create_thermodynamic_pointers(thermodynamic_variable, tracers,
    forcings, slow_forcings, right_hand_sides, intermediate_vars)
    key = generate_key(thermodynamic_variable)
    names = Tuple((key,))
    values = Tuple(((variable = thermodynamic_variable,
                     values = getproperty(tracers, key),
                     forcing = getproperty(forcings, key),
                     F = getproperty(slow_forcings, key),
                     R = getproperty(right_hand_sides, key),
                     IV = getproperty(intermediate_vars, key)),))
    return NamedTuple{names}(values)
end

function DryEarth(FT = Float64)
    return (ρ = EarthN₂O₂(FT),)
end

function DryEarth3(FT = Float64)
    return (ρ₁ = EarthN₂O₂(FT), ρ₂ = EarthN₂O₂(FT), ρ₃ = EarthN₂O₂(FT))
end

@inline generate_key(tvar::Entropy) = :ρs
@inline generate_key(tvar::Energy) = :ρe

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
    tracers = TracerFields(arch, grid, tracernames)
    return merge((ρu = ρu, ρv = ρv, ρw = ρw), tracers)
end

function RightHandSideFields(arch, grid, tracernames)
    ρu = FaceFieldX(arch, grid)
    ρv = FaceFieldY(arch, grid)
    ρw = FaceFieldZ(arch, grid)
    tracers = TracerFields(arch, grid, tracernames)
    return merge((ρu = ρu, ρv = ρv, ρw = ρw), tracers)
end
