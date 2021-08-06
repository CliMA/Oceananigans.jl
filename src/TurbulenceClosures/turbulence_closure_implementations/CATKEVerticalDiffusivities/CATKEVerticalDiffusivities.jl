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
                                diffusivity_scaling = RiDependentDiffusivityScaling{FT}(),
                                Cᴰ = 2.91,
                                mixing_length_parameter = 1.16,
                                surface_tke_flux = SurfaceTKEFlux{FT}(),
                                time_discretization::TD = ExplicitTimeDiscretization())

Returns the `CATKEVerticalDiffusivity` turbulence closure for vertical mixing by
small-scale ocean turbulence based on the prognostic evolution of subgrid
Turbulent Kinetic Energy (TKE).

`CATKEVerticalDiffusivity` is a downgradient, diffusive
closure formulated with three different eddy diffusivities for momentum, tracers, and TKE.
Each eddy diffusivity is the product of a diffusivity "scaling", a mixing length, and a turbulent
velocity scale which is the square root of the local TKE, such that

```math
Kᵠ = Cᵠ ℓ √e
```

where `Kᵠ` is the eddy diffusivity of `ϕ` where `ϕ` is either `u` (for momentum) `c` (for tracers), or
`e` (for TKE). `Cᵠ` is the diffusivity scaling for `ϕ`, `ℓ` is the mixing length
and `√e` is the turbulent velocity scale. The mixing length `ℓ` is modeled as

```math
ℓ = min(ℓᵇ, ℓᶻ)
```

where `ℓᵇ = Cᵇ * √e / N` and `ℓᶻ` is the distance to the nearest boundary.
`CATKEVerticalDiffusivity` also invokes a model for the flux of TKE across the numerical
ocean surface due to unstable buoyancy forcing and wind stress.

The `CATKEVerticalDiffusivity` is formulated in terms of 12 free parameters. These parameters
are _experimentally_ calibrated against large eddy simulations of ocean surface boundary layer turbulence
in idealized scenarios involving monotonic boundary layer deepening into variable stratification
due to constant surface momentum fluxes and/or destabilizing surface buoyancy flux.
This calibration has not been peer-reviewed, may be inaccurate and imperfect, and may not
be appropriate for three-dimensional ocean simulations.

See https://github.com/CliMA/LESbrary.jl for more information about the large eddy simulations.

The calibration procedure is not documented and is part of ongoing research.
The calibration was performed using a combination of Markov Chain Monte Carlo (MCMC)-based simulated
annealing and noisy Ensemble Kalman Inversion methods.

The one positional argument determines the floating point type of the free parameters
of `CATKEVerticalDiffusivity`. The default is `Float64`.

Keyword arguments
=================

* `diffusivity_scaling` : A group of parameters that scale the eddy diffusivity for momentum, tracers, and TKE.
                          The default is `RiDependentDiffusivityScaling{FT}()`, which represents a group of
                          parameters that implement a "smoothed step function" scaling that varies with the
                          local gradient Richardson number `Ri = ∂z(b) / (∂z(u)² + ∂z(v)²)`.

* `Cᴰ` : Parameter `Cᴰ` in the closure `ϵ = Cᴰ * e^3/2 / ℓ` that models the dissipation of TKE,
                            `ϵ`, appearing in the TKE evolution equation. The default is 2.91 via calibration
                            against large eddy simulations.
                          
* `mixing_length_parameter` : Parameter `Cᵇ` that multiplies the "buoyancy mixing length" `ℓᵇ = Cᵇ * √e / N`,
                            that appears in `CATKEVerticalDiffusivity`'s mixing length model.
                            The default is 1.16 via calibration against large eddy simulations.

* `time_discretization` : Either `ExplicitTimeDiscretization` or `VerticallyImplicitTimeDiscretization`.

"""
function CATKEVerticalDiffusivity(FT=Float64;
                                  Cᴰ = 2.91,
                                  mixing_length = MixingLength{FT}(),
                                  surface_tke_flux = SurfaceTKEFlux{FT}(),
                                  time_discretization::TD = VerticallyImplicitTimeDiscretization()) where TD

    @warn "CATKEVerticalDiffusivity is an experimental turbulence closure that \n" *
          "is unvalidated and whose default parameters are not calibrated for \n" * 
          "realistic ocean conditions or for use in a three-dimensional \n" *
          "simulation. Use with caution and report bugs and problems with physics \n" *
          "to https://github.com/CliMA/Oceananigans.jl/issues."

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
                    calculate_tke_diffusivities!,
                    diffusivities, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@kernel function calculate_tke_diffusivities!(diffusivities, grid, closure, velocities, tracers, buoyancy)
    i, j, k, = @index(Global, NTuple)
    @inbounds begin
        diffusivities.Kᵘ[i, j, k] = Kuᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
        diffusivities.Kᶜ[i, j, k] = Kcᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
        diffusivities.Kᵉ[i, j, k] = Keᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
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
