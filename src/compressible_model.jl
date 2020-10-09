using Oceananigans
using Oceananigans.Models: AbstractModel, Clock, tracernames
using Oceananigans.Forcings: model_forcing

#####
##### Definition of a compressible model
#####

mutable struct CompressibleModel{A, FT, Ω, D, M, V, T, L, K, Θ, G, X, C, F, S, R, I} <: AbstractModel
              architecture :: A
                      grid :: Ω
                     clock :: Clock{FT}
             total_density :: D
                   momenta :: M
                velocities :: V
                   tracers :: T
              lazy_tracers :: L
             diffusivities :: K
    thermodynamic_variable :: Θ
                     gases :: G
                   gravity :: FT
                  coriolis :: X
                   closure :: C
                   forcing :: F
         slow_source_terms :: S
         fast_source_terms :: R
       intermediate_fields :: I
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
               tracernames = collect_tracers(thermodynamic_variable, gases),
                  coriolis = nothing,
                   closure = IsotropicDiffusivity(float_type, ν=0.5, κ=0.5),
       boundary_conditions = NamedTuple(),
             diffusivities = DiffusivityFields(architecture, grid, tracernames, boundary_conditions, closure),
                   forcing = NamedTuple(),
                   gravity = g_Earth,
         slow_source_terms = SlowSourceTermFields(architecture, grid, tracernames),
         fast_source_terms = FastSourceTermFields(architecture, grid, tracernames),
       intermediate_fields = FastSourceTermFields(architecture, grid, tracernames))

    gravity = float_type(gravity)
    tracers = TracerFields(architecture, grid, tracernames)
    forcing = model_forcing(tracernames; forcing...)
    closure = with_tracers(tracernames, closure)
    total_density = CellField(architecture, grid)

    velocities = LazyVelocityFields(architecture, grid, total_density, momenta)
    lazy_tracers = LazyTracerFields(architecture, grid, total_density, tracers)

    return CompressibleModel(
        architecture, grid, clock, total_density, momenta, velocities, tracers,
        lazy_tracers, diffusivities, thermodynamic_variable, gases, gravity,
        coriolis, closure, forcing, slow_source_terms, fast_source_terms,
        intermediate_fields)
end

using Oceananigans.Grids: short_show

Base.show(io::IO, model::CompressibleModel{A, FT}) where {A, FT} =
    print(io, "CompressibleModel{$A, $FT} with $(length(model.gases)) gas(es) ",
        "(time = $(prettytime(model.clock.time)), iteration = $(model.clock.iteration)) \n",
        "├── grid: $(short_show(model.grid))\n",
        "├── tracers: $(tracernames(model.tracers))\n",
        "├── closure: $(typeof(model.closure))\n",
        "└── coriolis: $(typeof(model.coriolis))\n")

#####
##### Utilities for constructing compressible models
#####

generate_key(::Entropy) = :ρs
generate_key(::Energy) = :ρe

function collect_tracers(thermodynamic_variable, args...)
    list = [generate_key(thermodynamic_variable)]
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

function DryEarth3(FT = Float64)
    return (ρ₁ = EarthN₂O₂(FT), ρ₂ = EarthN₂O₂(FT), ρ₃ = EarthN₂O₂(FT))
end

function MomentumFields(arch, grid)
    ρu = XFaceField(arch, grid)
    ρv = YFaceField(arch, grid)
    ρw = ZFaceField(arch, grid)
    return (ρu=ρu, ρv=ρv, ρw=ρw)
end

function TracerFields(arch, grid, tracernames)
    tracerfields = Tuple(CellField(arch, grid) for c in tracernames)
    return NamedTuple{tracernames}(tracerfields)
end

function SlowSourceTermFields(arch, grid, tracernames)
    ρu = XFaceField(arch, grid)
    ρv = YFaceField(arch, grid)
    ρw = ZFaceField(arch, grid)
    tracers = TracerFields(arch, grid, tracernames)
    return (ρu = ρu, ρv = ρv, ρw = ρw, tracers = tracers)
end

function FastSourceTermFields(arch, grid, tracernames)
    ρu = XFaceField(arch, grid)
    ρv = YFaceField(arch, grid)
    ρw = ZFaceField(arch, grid)
    tracers = TracerFields(arch, grid, tracernames)
    return (ρu = ρu, ρv = ρv, ρw = ρw, tracers = tracers)
end
