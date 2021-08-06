module CATKEVerticalDiffusivities

using Adapt
using KernelAbstractions: @kernel, @index

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.BoundaryConditions: default_prognostic_field_boundary_condition
using Oceananigans.BoundaryConditions: BoundaryCondition, FieldBoundaryConditions, DiscreteBoundaryFunction
using Oceananigans.BuoyancyModels: ∂z_b, top_buoyancy_flux
using Oceananigans.Operators: ℑzᵃᵃᶜ

using Oceananigans.TurbulenceClosures:
    AbstractTurbulenceClosure,
    ExplicitTimeDiscretization,
    VerticallyImplicitTimeDiscretization

import Oceananigans.BoundaryConditions: getbc
import Oceananigans.Utils: with_tracers
import Oceananigans.TurbulenceClosures: calculate_diffusivities!

function hydrostatic_turbulent_kinetic_energy_tendency end

struct CATKEVerticalDiffusivity{TD, CD, CL, CQ} <: AbstractTurbulenceClosure{TD}
    Cᴰ :: CD
    mixing_length :: CL
    surface_tke_flux :: CQ
end

function CATKEVerticalDiffusivity{TD}(Cᴰ:: CD,
                                      mixing_length :: CL,
                                      surface_tke_flux :: CQ) where {TD, CD, CL, CQ}

    return CATKEVerticalDiffusivity{TD, CD, CL, CQ}(Cᴰ, mixing_length, surface_tke_flux)
end

const CATKEVD = CATKEVerticalDiffusivity

# Support for "ManyIndependentColumnMode"
const CATKEVDArray = AbstractArray{<:CATKEVD}

function with_tracers(tracer_names, closure::Union{CATKEVD, CATKEVDArray})
    :e ∈ tracer_names || error("Tracers must contain :e to represent turbulent kinetic energy for `CATKEVerticalDiffusivity`.")
    return closure
end

include("mixing_length.jl")
include("surface_TKE_flux.jl")
include("turbulent_kinetic_energy_equation.jl")

for S in (:MixingLength, :SurfaceTKEFlux)
    @eval @inline convert_eltype(::Type{FT}, s::$S) where FT = $S{FT}(; Dict(p => getproperty(s, p) for p in propertynames(s))...)
    @eval @inline convert_eltype(::Type{FT}, s::$S{FT}) where FT = s
end

"""
    CATKEVerticalDiffusivity(FT=Float64;
                             Cᴰ = 2.91,
                             mixing_length = MixingLength{FT}(),
                             surface_tke_flux = SurfaceTKEFlux{FT}(),
                             time_discretization::TD = ExplicitTimeDiscretization())

Returns the `CATKEVerticalDiffusivity` turbulence closure for vertical mixing by
small-scale ocean turbulence based on the prognostic evolution of subgrid
Turbulent Kinetic Energy (TKE).
"""
function CATKEVerticalDiffusivity(FT=Float64;
                                  Cᴰ = 2.91,
                                  mixing_length = MixingLength{FT}(),
                                  surface_tke_flux = SurfaceTKEFlux{FT}(),
                                  warning = true,
                                  time_discretization::TD = VerticallyImplicitTimeDiscretization()) where TD

    if warning
        @warn "CATKEVerticalDiffusivity is an experimental turbulence closure that \n" *
              "is unvalidated and whose default parameters are not calibrated for \n" * 
              "realistic ocean conditions or for use in a three-dimensional \n" *
              "simulation. Use with caution and report bugs and problems with physics \n" *
              "to https://github.com/CliMA/Oceananigans.jl/issues."
    end

    Cᴰ = convert(FT, Cᴰ)
    mixing_length = convert_eltype(FT, mixing_length)
    surface_tke_flux = convert_eltype(FT, surface_tke_flux)

    return CATKEVerticalDiffusivity{TD}(Cᴰ, mixing_length, surface_tke_flux)
end

#####
##### Diffusivities and diffusivity fields utilities
#####

function DiffusivityFields(arch, grid, tracer_names, bcs, closure::Union{CATKEVD, CATKEVDArray})

    default_diffusivity_bcs = (Kᵘ = FieldBoundaryConditions(grid, (Center, Center, Center)),
                               Kᶜ = FieldBoundaryConditions(grid, (Center, Center, Center)),
                               Kᵉ = FieldBoundaryConditions(grid, (Center, Center, Center)))

    bcs = merge(default_diffusivity_bcs, bcs)

    Kᵘ = CenterField(arch, grid, bcs.Kᵘ)
    Kᶜ = CenterField(arch, grid, bcs.Kᶜ)
    Kᵉ = CenterField(arch, grid, bcs.Kᵉ)

    return (; Kᵘ, Kᶜ, Kᵉ)
end        

function calculate_diffusivities!(diffusivities, closure::CATKEVD, model)

    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    clock = model.clock
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    event = launch!(arch, grid, :xyz,
                    calculate_CATKE_diffusivities!,
                    diffusivities, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@kernel function calculate_CATKE_diffusivities!(diffusivities, grid, args...)
    i, j, k, = @index(Global, NTuple)
    @inbounds begin
        diffusivities.Kᵘ[i, j, k] = Kuᶜᶜᶜ(i, j, k, grid, closure, args...)
        diffusivities.Kᶜ[i, j, k] = Kcᶜᶜᶜ(i, j, k, grid, closure, args...)
        diffusivities.Kᵉ[i, j, k] = Keᶜᶜᶜ(i, j, k, grid, closure, args...)
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

@inline function viscous_flux_uz(i, j, k, grid, closure::CATKEVD, clock, velocities, diffusivities, args...)
    Ku = ℑxzᶠᵃᶠ(i, j, k, grid, diffusivities.Kᵘ)
    return - Ku * ∂zᵃᵃᶠ(i, j, k, grid, velocities.u)
end

@inline function viscous_flux_vz(i, j, k, grid, closure::CATKEVD, clock, velocities, diffusivities, args...)
    Kv = ℑyzᵃᶠᶠ(i, j, k, grid, diffusivities.Kᵘ)
    return - Kv * ∂zᵃᵃᶠ(i, j, k, grid, velocities.v)
end

@inline function viscous_flux_wz(i, j, k, grid, closure::CATKEVD, clock, velocities, diffusivities, args...)
    @inbounds Kw = diffusivities.Kᵘ[i, j, k]
    return - Kw * ∂zᵃᵃᶜ(i, j, k, grid, velocities.w)
end

@inline function diffusive_flux_z(i, j, k, grid, closure::CATKEVD, c, tracer_index, clock, diffusivities, args...)
    Kcᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, diffusivities.Kᶜ)
    return - Kcᶜᶜᶠ * ∂zᵃᵃᶠ(i, j, k, grid, c)
end

# Diffusive flux of TKE!
@inline function diffusive_flux_z(i, j, k, grid, closure::CATKEVD, e, ::TKETracerIndex, clock, diffusivities, args...)
    Keᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, diffusivities.Kᵉ)
    return - Keᶜᶜᶠ * ∂zᵃᵃᶠ(i, j, k, grid, e)
end

# "Translations" for diffusive transport by non-CATKEVD closures
@inline diffusive_flux_x(i, j, k, grid, closure, e, ::TKETracerIndex{N}, args...) where N = diffusive_flux_x(i, j, k, grid, closure, e, Val(N), args...)
@inline diffusive_flux_y(i, j, k, grid, closure, e, ::TKETracerIndex{N}, args...) where N = diffusive_flux_y(i, j, k, grid, closure, e, Val(N), args...)
@inline diffusive_flux_z(i, j, k, grid, closure, e, ::TKETracerIndex{N}, args...) where N = diffusive_flux_z(i, j, k, grid, closure, e, Val(N), args...)

# Shortcuts --- CATKEVD incurs no horizontal transport
@inline viscous_flux_ux(i, j, k, grid, ::CATKEVD, args...) = zero(eltype(grid))
@inline viscous_flux_uy(i, j, k, grid, ::CATKEVD, args...) = zero(eltype(grid))
@inline viscous_flux_vx(i, j, k, grid, ::CATKEVD, args...) = zero(eltype(grid))
@inline viscous_flux_vy(i, j, k, grid, ::CATKEVD, args...) = zero(eltype(grid))
@inline viscous_flux_wx(i, j, k, grid, ::CATKEVD, args...) = zero(eltype(grid))
@inline viscous_flux_wy(i, j, k, grid, ::CATKEVD, args...) = zero(eltype(grid))
@inline diffusive_flux_x(i, j, k, grid, ::CATKEVD, args...) = zero(eltype(grid))
@inline diffusive_flux_y(i, j, k, grid, ::CATKEVD, args...) = zero(eltype(grid))

# Disambiguate
@inline diffusive_flux_x(i, j, k, grid, ::CATKEVD, e, ::TKETracerIndex, args...) = zero(eltype(grid))
@inline diffusive_flux_y(i, j, k, grid, ::CATKEVD, e, ::TKETracerIndex, args...) = zero(eltype(grid))

#####
##### Support for VerticallyImplicitTimeDiscretization
#####

const VITD = VerticallyImplicitTimeDiscretization

@inline z_viscosity(closure::Union{CATKEVD, CATKEVDArray}, diffusivities, args...) = diffusivities.Kᵘ

@inline function z_diffusivity(closure::Union{CATKEVD, CATKEVDArray}, ::Val{tracer_index},
                               diffusivities, tracers, args...) where tracer_index

    tke_index = findfirst(name -> name === :e, keys(tracers))

    if tracer_index === tke_index
        return diffusivities.Kᵉ
    else
        return diffusivities.Kᶜ
    end
end

const VerticallyBoundedGrid{FT} = AbstractGrid{FT, <:Any, <:Any, <:Bounded}

@inline diffusive_flux_z(i, j, k, grid, ::VITD, closure::CATKEVD, args...) = zero(eltype(grid))
@inline viscous_flux_uz(i, j, k, grid, ::VITD, closure::CATKEVD, args...) = zero(eltype(grid))
@inline viscous_flux_vz(i, j, k, grid, ::VITD, closure::CATKEVD, args...) = zero(eltype(grid))

@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::CATKEVD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

@inline function viscous_flux_uz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::CATKEVD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_uz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

@inline function viscous_flux_vz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::CATKEVD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_vz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

@inline function viscous_flux_wz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::CATKEVD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_wz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

end
