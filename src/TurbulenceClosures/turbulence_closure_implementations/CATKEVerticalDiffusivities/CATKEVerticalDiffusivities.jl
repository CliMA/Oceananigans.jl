module CATKEVerticalDiffusivities

using Adapt
using KernelAbstractions: @kernel, @index

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Utils
using Oceananigans.Fields
using Oceananigans.BoundaryConditions: default_prognostic_field_boundary_condition
using Oceananigans.BoundaryConditions: BoundaryCondition, FieldBoundaryConditions, DiscreteBoundaryFunction, FluxBoundaryCondition
using Oceananigans.BuoyancyModels: ∂z_b, top_buoyancy_flux
using Oceananigans.Operators: ℑzᵃᵃᶜ

using Oceananigans.TurbulenceClosures:
    get_closure_ij,
    AbstractTurbulenceClosure,
    AbstractScalarDiffusivity,
    ExplicitTimeDiscretization,
    VerticallyImplicitTimeDiscretization,
    ThreeDimensionalFormulation, 
    VerticalFormulation

import Oceananigans.BoundaryConditions: getbc
import Oceananigans.Utils: with_tracers
import Oceananigans.TurbulenceClosures:
    validate_closure,
    add_closure_specific_boundary_conditions,
    calculate_diffusivities!,
    DiffusivityFields,
    viscosity,
    diffusivity,
    diffusive_flux_x,
    diffusive_flux_y,
    diffusive_flux_z

function hydrostatic_turbulent_kinetic_energy_tendency end

struct CATKEVerticalDiffusivity{TD, CD, CL, CQ} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    Cᴰ :: CD
    mixing_length :: CL
    surface_TKE_flux :: CQ
end

function CATKEVerticalDiffusivity{TD}(Cᴰ:: CD,
                                      mixing_length :: CL,
                                      surface_TKE_flux :: CQ) where {TD, CD, CL, CQ}

    return CATKEVerticalDiffusivity{TD, CD, CL, CQ}(Cᴰ, mixing_length, surface_TKE_flux)
end

const CATKEVD = CATKEVerticalDiffusivity

# Support for "ManyIndependentColumnMode"
const CATKEVDArray = AbstractArray{<:CATKEVD}
const FlavorOfCATKE = Union{CATKEVD, CATKEVDArray}

function with_tracers(tracer_names, closure::FlavorOfCATKE)
    :e ∈ tracer_names ||
        throw(ArgumentError("Tracers must contain :e to represent turbulent kinetic energy " *
                            "for `CATKEVerticalDiffusivity`."))

    return closure
end

# Closure tuple sorting
validate_closure(closure_tuple::Tuple) = Tuple(sort(collect(closure_tuple), lt=catke_first))

catke_first(closure1, catke::FlavorOfCATKE) = false
catke_first(catke::FlavorOfCATKE, closure2) = true
catke_first(closure1, closure2) = true

# Two CATKEs?!
catke_first(catke1::FlavorOfCATKE, catke2::FlavorOfCATKE) = error("Can't have two CATKEs in one closure tuple.")

include("mixing_length.jl")
include("surface_TKE_flux.jl")
include("turbulent_kinetic_energy_equation.jl")

for S in (:MixingLength, :SurfaceTKEFlux)
    @eval @inline convert_eltype(::Type{FT}, s::$S) where FT = $S{FT}(; Dict(p => getproperty(s, p) for p in propertynames(s))...)
    @eval @inline convert_eltype(::Type{FT}, s::$S{FT}) where FT = s
end

"""
    CATKEVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization, FT=Float64;
                             Cᴰ = 2.91,
                             mixing_length = MixingLength{FT}(),
                             surface_TKE_flux = SurfaceTKEFlux{FT}(),
                             warning = true)

Returns the `CATKEVerticalDiffusivity` turbulence closure for vertical mixing by
small-scale ocean turbulence based on the prognostic evolution of subgrid
Turbulent Kinetic Energy (TKE).
"""
CATKEVerticalDiffusivity(FT::DataType; kwargs...) = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kwargs...)

function CATKEVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(), FT=Float64;
                                  Cᴰ = 2.91,
                                  mixing_length = MixingLength{FT}(),
                                  surface_TKE_flux = SurfaceTKEFlux{FT}(),
                                  warning = true)

    if warning
        @warn "CATKEVerticalDiffusivity is an experimental turbulence closure that \n" *
              "is unvalidated and whose default parameters are not calibrated for \n" * 
              "realistic ocean conditions or for use in a three-dimensional \n" *
              "simulation. Use with caution and report bugs and problems with physics \n" *
              "to https://github.com/CliMA/Oceananigans.jl/issues."
    end

    Cᴰ = convert(FT, Cᴰ)
    mixing_length = convert_eltype(FT, mixing_length)
    surface_TKE_flux = convert_eltype(FT, surface_TKE_flux)

    return CATKEVerticalDiffusivity{typeof(time_discretization)}(Cᴰ, mixing_length, surface_TKE_flux)
end

#####
##### Diffusivities and diffusivity fields utilities
#####

function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfCATKE)

    default_diffusivity_bcs = (Kᵘ = FieldBoundaryConditions(grid, (Center, Center, Center)),
                               Kᶜ = FieldBoundaryConditions(grid, (Center, Center, Center)),
                               Kᵉ = FieldBoundaryConditions(grid, (Center, Center, Center)))

    bcs = merge(default_diffusivity_bcs, bcs)

    Kᵘ = CenterField(grid, boundary_conditions=bcs.Kᵘ)
    Kᶜ = CenterField(grid, boundary_conditions=bcs.Kᶜ)
    Kᵉ = CenterField(grid, boundary_conditions=bcs.Kᵉ)

    return (; Kᵘ, Kᶜ, Kᵉ)
end        

function calculate_diffusivities!(diffusivities, closure::FlavorOfCATKE, model)

    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy
    clock = model.clock
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    event = launch!(arch, grid, :xyz,
                    calculate_CATKE_diffusivities!,
                    diffusivities, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@kernel function calculate_CATKE_diffusivities!(diffusivities, grid, closure::FlavorOfCATKE, args...)
    i, j, k, = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = get_closure_ij(i, j, closure)

    @inbounds begin
        diffusivities.Kᵘ[i, j, k] = Kuᶜᶜᶜ(i, j, k, grid, closure_ij, args...)
        diffusivities.Kᶜ[i, j, k] = Kcᶜᶜᶜ(i, j, k, grid, closure_ij, args...)
        diffusivities.Kᵉ[i, j, k] = Keᶜᶜᶜ(i, j, k, grid, closure_ij, args...)
    end
end

@inline turbulent_velocity(i, j, k, grid, e) = @inbounds sqrt(max(zero(eltype(grid)), e[i, j, k]))

@inline function Kuᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    u★ = turbulent_velocity(i, j, k, grid, tracers.e)
    ℓu = momentum_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    return ℓu * u★
end

@inline function Kcᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    u★ = turbulent_velocity(i, j, k, grid, tracers.e)
    ℓc = tracer_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    return ℓc * u★
end

@inline function Keᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    u★ = turbulent_velocity(i, j, k, grid, tracers.e)
    ℓe = TKE_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    return ℓe * u★
end

#####
##### Viscous flux, diffusive fluxes, plus shenanigans for diffusive fluxes of TKE (eg TKE "transport")
#####

# Special "index type" alternative to Val for dispatch
struct TKETracerIndex{N} end

@inline TKETracerIndex(N) = TKETracerIndex{N}()

# Diffusive flux of TKE!
@inline function diffusive_flux_z(i, j, k, grid, closure::CATKEVD, e, ::TKETracerIndex, clock, diffusivities, args...)
    Keᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, diffusivities.Kᵉ)
    return - Keᶜᶜᶠ * ∂zᶜᶜᶠ(i, j, k, grid, e)
end

# "Translations" for diffusive transport by non-CATKEVD closures
const ATC = AbstractTurbulenceClosure
@inline diffusive_flux_x(i, j, k, grid, clo::ATC, e, ::TKETracerIndex{N}, args...) where N = diffusive_flux_x(i, j, k, grid, clo, e, Val(N), args...)
@inline diffusive_flux_y(i, j, k, grid, clo::ATC, e, ::TKETracerIndex{N}, args...) where N = diffusive_flux_y(i, j, k, grid, clo, e, Val(N), args...)
@inline diffusive_flux_z(i, j, k, grid, clo::ATC, e, ::TKETracerIndex{N}, args...) where N = diffusive_flux_z(i, j, k, grid, clo, e, Val(N), args...)

@inline viscosity(::FlavorOfCATKE, diffusivities, args...) = diffusivities.Kᵘ

@inline function diffusivity(::FlavorOfCATKE, ::Val{tracer_index},
                               diffusivities, tracers, args...) where tracer_index

    tke_index = findfirst(name -> name === :e, keys(tracers))

    if tracer_index === tke_index
        return diffusivities.Kᵉ
    else
        return diffusivities.Kᶜ
    end
end

#####
##### Show
#####

function Base.summary(closure::CATKEVD)
    TD = nameof(typeof(time_discretization(closure)))
    return string("CATKEVerticalDiffusivity{$TD}")
end

Base.show(io::IO, closure::FlavorOfCATKE) = print(io, summary(closure))

