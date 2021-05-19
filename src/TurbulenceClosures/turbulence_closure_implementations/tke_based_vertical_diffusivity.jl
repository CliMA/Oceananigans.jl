using Oceananigans.Architectures: architecture, device_event
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators: ℑzᵃᵃᶜ

struct TKEBasedVerticalDiffusivity{TD, CK, CD, CL, CQ} <: AbstractTurbulenceClosure{TD}
    diffusivity_scaling :: CK
    dissipation_parameter :: CD
    mixing_length_parameter :: CL
    surface_model :: CQ

    function TKEBasedVerticalDiffusivity{TD}(
        diffusivity_scaling :: CK,
        dissipation_parameter :: CD,
        mixing_length_parameter :: CL,
        surface_model :: CQ) where {TD, CK, CD, CL, CQ}

        return new{TD, CK, CD, CL, CQ}(diffusivity_scaling, dissipation_parameter, mixing_length_parameter, surface_model)
    end
end

"""
    TKEBasedVerticalDiffusivity <: AbstractTurbulenceClosure{TD}

Parameters for the "anisotropic minimum dissipation" turbulence closure for large eddy simulation
proposed originally by [Rozema15](@cite) and [Abkar16](@cite), and then modified
by [Verstappen18](@cite), and finally described and validated for by [Vreugdenhil18](@cite).
"""
function TKEBasedVerticalDiffusivity(FT=Float64;
                                     diffusivity_scaling = RiDependentDiffusivityScaling(),
                                     dissipation_parameter = 2.91,
                                     mixing_length_parameter = 1.16,
                                     surface_model = TKESurfaceFlux(),
                                     time_discretization::TD = ExplicitTimeDiscretization()) where TD

    return TKEBasedVerticalDiffusivity{TD}(diffusivity_scaling,
                                           dissipation_parameter,
                                           mixing_length_parameter,
                                           surface_model)
end

const TKEVD = TKEBasedVerticalDiffusivity

"""
    struct RiDependentDiffusivityScaling{FT}

A diffusivity model in which momentum, tracers, and TKE
each have Richardson-number-dependent diffusivities.
The Richardson number is

    ``Ri = ∂z B / ( (∂z U)² + (∂z V)² )`` ,

where ``B`` is buoyancy and ``∂z`` denotes a vertical derviative.
The Richardson-number dependent diffusivities are multiplied by the stability
function

    ``σ(Ri) = σ⁻ + (σ⁺ - σ⁻) * step(Ri, Riᶜ, Riʷ)``

where ``σ⁰``, ``σᵟ``, ``Riᶜ``, and ``Riʷ`` are free parameters,
and ``step`` is a smooth step function defined by

    ``step(x, c, w) = (1 + tanh((x - c) / w)) / 2``.
"""
Base.@kwdef struct RiDependentDiffusivityScaling{FT}
    Cᴷu⁻  :: FT = 0.15
    Cᴷu⁺  :: FT = 0.73
    Cᴷc⁻  :: FT = 0.40
    Cᴷc⁺  :: FT = 1.77
    Cᴷe⁻  :: FT = 0.13
    Cᴷe⁺  :: FT = 1.22
    CᴷRiʷ :: FT = 0.72
    CᴷRiᶜ :: FT = 0.76
end

Base.@kwdef struct TKESurfaceFlux{FT}
    Cᵂu★ :: FT = 3.62
    CᵂwΔ :: FT = 1.31
end

#####
##### Utilities
#####

#
# TODO: figure out how to calculate the current buoyancy flux and momentum flux...
# function validate_closure_dependent_boundary_conditions(closure::TKEVD, boundary_conditions)
#
# end

function DiffusivityFields(arch, grid, tracer_names, bcs, closure::TKEVD)

    Kᵘ_bcs = :Kᵘ ∈ keys(bcs) ? bcs[:Kᵘ] : DiffusivityBoundaryConditions(grid)
    Kᶜ_bcs = :Kᶜ ∈ keys(bcs) ? bcs[:Kᶜ] : DiffusivityBoundaryConditions(grid)
    Kᵉ_bcs = :Kᵉ ∈ keys(bcs) ? bcs[:Kᵉ] : DiffusivityBoundaryConditions(grid)

    Kᵘ = CenterField(arch, grid, Kᵘ_bcs)
    Kᶜ = CenterField(arch, grid, Kᶜ_bcs)
    Kᵉ = CenterField(arch, grid, Kᵉ_bcs)

    return (; Kᵘ, Kᶜ, Kᵉ)
end        
            
function with_tracers(tracer_names, closure::TKEVD)
    :e ∈ tracer_names || error("Tracers must contain :e to represent turbulent kinetic energy for `TKEBasedVerticalDiffusivity`.")
    return closure
end

function calculate_diffusivities!(diffusivities, arch, grid, closure::TKEVD, buoyancy, velocities, tracers)

    e = tracers.e

    event = launch!(arch, grid, :xyz,
                    calculate_tke_diffusivities!, diffusivities, grid, closure, e, velocities, tracers, buoyancy,
                    dependencies=device_event(arch))

    wait(device(arch), event)

    return nothing
end

@kernel function calculate_tke_diffusivities!(diffusivities, grid, closure, e, velocities, tracers, buoyancy)
    i, j, k, = @index(Global, NTuple)
    @inbounds begin
        diffusivities.Kᵘ[i, j, k] = Kuᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
        diffusivities.Kᶜ[i, j, k] = Kcᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
        diffusivities.Kᵉ[i, j, k] = Keᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    end
end

function hydrostatic_turbulent_kinetic_energy_tendency end

#####
##### Mixing length
#####

@inline surface(i, j, k, grid)                = znode(Center(), Center(), Face(), i, j, grid.Nz+1, grid)
@inline bottom(i, j, k, grid)                 = znode(Center(), Center(), Face(), i, j, 1, grid)
@inline depthᶜᶜᶜ(i, j, k, grid)               = surface(i, j, k, grid) - znode(Center(), Center(), Center(), i, j, k, grid)
@inline height_above_bottomᶜᶜᶜ(i, j, k, grid) = znode(Center(), Center(), Center(), i, j, k, grid) - bottom(i, j, k, grid)

@inline wall_vertical_distanceᶜᶜᶜ(i, j, k, grid) = min(depthᶜᶜᶜ(i, j, k, grid), height_above_bottomᶜᶜᶜ(i, j, k, grid))

@inline function sqrt_∂z_b(i, j, k, grid, buoyancy, tracers)
    FT = eltype(grid)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    N²⁺ = max(zero(FT), N²)
    return sqrt(N²⁺)  
end

@inline function buoyancy_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    FT = eltype(grid)
    Cᵇ = closure.mixing_length_parameter
    N⁺ = ℑzᵃᵃᶜ(i, j, k, grid, sqrt_∂z_b, buoyancy, tracers)

    @inbounds e⁺ = max(zero(FT), e[i, j, k])

    return @inbounds ifelse(N⁺ == 0, FT(Inf), Cᵇ * sqrt(e⁺) / N⁺)
end

@inline function dissipation_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    ℓᶻ = wall_vertical_distanceᶜᶜᶜ(i, j, k, grid)
    ℓᵇ = buoyancy_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    ℓ = min(ℓᶻ, ℓᵇ)
    ℓ_min = Δzᵃᵃᶜ(i, j, k, grid) / 2 # minimum mixing length...
    return max(ℓ_min, ℓ)
end

#####
##### Diffusivities
#####

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    FT = eltype(grid)
    ∂z_u² = ℑxzᶜᵃᶜ(i, j, k, grid, ϕ², ∂zᵃᵃᶠ, velocities.u)
    ∂z_v² = ℑyzᵃᶜᶜ(i, j, k, grid, ϕ², ∂zᵃᵃᶠ, velocities.v)
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    return ifelse(N² == 0, zero(FT), N² / (∂z_u² + ∂z_v²))
end

@inline step(x, c, w) = (1 + tanh((x - c) / w)) / 2

@inline scale(Ri, σ⁻, σ⁺, c, w) = σ⁻ + (σ⁺ - σ⁻) * step(Ri, c, w)

@inline function momentum_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    return scale(Ri,
                 closure.diffusivity_scaling.Cᴷu⁻,
                 closure.diffusivity_scaling.Cᴷu⁺,
                 closure.diffusivity_scaling.CᴷRiᶜ,
                 closure.diffusivity_scaling.CᴷRiʷ)
end

@inline function tracer_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    return scale(Ri,
                 closure.diffusivity_scaling.Cᴷc⁻,
                 closure.diffusivity_scaling.Cᴷc⁺,
                 closure.diffusivity_scaling.CᴷRiᶜ,
                 closure.diffusivity_scaling.CᴷRiʷ)
end

@inline function TKE_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    return scale(Ri,
                 closure.diffusivity_scaling.Cᴷe⁻,
                 closure.diffusivity_scaling.Cᴷe⁺,
                 closure.diffusivity_scaling.CᴷRiᶜ,
                 closure.diffusivity_scaling.CᴷRiʷ)
end

@inline turbulent_velocity(i, j, k, grid, e) = @inbounds sqrt(max(zero(eltype(grid)), e[i, j, k]))

@inline function unscaled_eddy_diffusivityᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    ℓ = dissipation_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    u★ = turbulent_velocity(i, j, k, grid, e)
    return ℓ * u★
end

@inline function Kuᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    K = unscaled_eddy_diffusivityᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    σu = momentum_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    return σu * K
end

@inline function Kcᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    K = unscaled_eddy_diffusivityᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    σc = tracer_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    return σc * K
end

@inline function Keᶜᶜᶜ(i, j, k, grid, closure, e, velocities, tracers, buoyancy)
    K = unscaled_eddy_diffusivityᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    σe = TKE_diffusivity_scale(i, j, k, grid, closure, velocities, tracers, buoyancy)
    return σe * K
end

#####
##### Terms in the turbulent kinetic energy equation, all at cell centers
#####

@inline ϕ²(i, j, k, grid, ϕ) = ϕ(i, j, k, grid)^2

@inline function shear_production(i, j, k, grid, closure, clock, velocities, tracers, buoyancy, diffusivities)
    ∂z_u² = ℑxzᶜᵃᶜ(i, j, k, grid, ϕ², ∂zᵃᵃᶠ, velocities.u)
    ∂z_v² = ℑyzᵃᶜᶜ(i, j, k, grid, ϕ², ∂zᵃᵃᶠ, velocities.v)
    Ku = Kuᶜᶜᶜ(i, j, k, grid, closure, tracers.e, velocities, tracers, buoyancy)
    return Ku * (∂z_u² + ∂z_v²)
end

@inline function buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy)
    Kc = Kcᶜᶜᶜ(i, j, k, grid, closure, tracers.e, velocities, tracers, buoyancy)
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    return - Kc * N²
end

@inline function dissipation(i, j, k, grid, closure, tracers, buoyancy)
    e = tracers.e
    FT = eltype(grid)
    three_halves = FT(3/2)
    @inbounds ẽ³² = abs(e[i, j, k])^three_halves

    ℓ = dissipation_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, e, tracers, buoyancy)
    Cᴰ = closure.dissipation_parameter

    return Cᴰ * ẽ³² / ℓ
end

#####
##### Viscous flux, diffusive fluxes, plus shenanigans for diffusive fluxes of TKE (eg TKE "transport")
#####

# Special "index type" alternative to Val for dispatch
struct TKETracerIndex{N} end
@inline TKETracerIndex(N) = TKETracerIndex{N}()

@inline function viscous_flux_uz(i, j, k, grid, closure::TKEVD, clock, velocities, diffusivities, tracers, buoyancy)
    Kuᶠᶜᶠ = ℑxzᶠᵃᶠ(i, j, k, grid, diffusivities.Kᵘ)
    return - Kuᶠᶜᶠ * ∂zᵃᵃᶠ(i, j, k, grid, velocities.u)
end

@inline function viscous_flux_vz(i, j, k, grid, closure::TKEVD, clock, velocities, diffusivities, tracers, buoyancy)
    Kuᶜᶠᶠ = ℑyzᵃᶠᶠ(i, j, k, grid, diffusivities.Kᵘ)
    return - Kuᶜᶠᶠ * ∂zᵃᵃᶠ(i, j, k, grid, velocities.v)
end

@inline function diffusive_flux_z(i, j, k, grid, closure::TKEVD, c, tracer_index, clock, diffusivities, tracers, buoyancy, velocities)
    Kcᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, diffusivities.Kᶜ)
    return - Kcᶜᶜᶠ * ∂zᵃᵃᶠ(i, j, k, grid, c)
end

# Diffusive flux of TKE!
@inline function diffusive_flux_z(i, j, k, grid, closure::TKEVD, e, ::TKETracerIndex, clock, diffusivities, tracers, buoyancy, velocities)
    Keᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, diffusivities.Kᵉ)
    return - Keᶜᶜᶠ * ∂zᵃᵃᶠ(i, j, k, grid, e)
end

# "Translations" for diffusive transport by non-TKEVD closures
@inline diffusive_flux_x(i, j, k, grid, closure, e, ::TKETracerIndex{N}, args...) where N = diffusive_flux_x(i, j, k, grid, closure, e, Val(N), args...)
@inline diffusive_flux_y(i, j, k, grid, closure, e, ::TKETracerIndex{N}, args...) where N = diffusive_flux_y(i, j, k, grid, closure, e, Val(N), args...)
@inline diffusive_flux_z(i, j, k, grid, closure, e, ::TKETracerIndex{N}, args...) where N = diffusive_flux_z(i, j, k, grid, closure, e, Val(N), args...)

# Shortcuts --- TKEVD incurs no horizontal transport
@inline diffusive_flux_x(i, j, k, grid, ::TKEVD, args...) = zero(eltype(grid))
@inline diffusive_flux_y(i, j, k, grid, ::TKEVD, args...) = zero(eltype(grid))

#####
##### Support for VerticallyImplicitTimeDiscretization
#####

const VITD = VerticallyImplicitTimeDiscretization

@inline z_viscosity(closure::TKEVD, diffusivities, velocities, tracers, buoyancy) = diffusivities.Kᵘ

@inline function z_diffusivity(closure::TKEVD, ::Val{tracer_index}, diffusivities, velocities, tracers, buoyancy) where tracer_index
    tke_index = findfirst(name -> name === :e, keys(tracers))

    if tracer_index === tke_index
        return diffusivities.Kᵉ
    else
        return diffusivities.Kᶜ
    end
end

const VerticallyBoundedGrid{FT} = AbstractPrimaryGrid{FT, <:Any, <:Any, <:Bounded}

@inline diffusive_flux_z(i, j, k, grid::APG{FT}, ::VITD, closure::TKEVD, args...) where FT = zero(FT)
@inline viscous_flux_uz(i, j, k, grid::APG{FT}, ::VITD, closure::TKEVD, args...) where FT = zero(FT)
@inline viscous_flux_vz(i, j, k, grid::APG{FT}, ::VITD, closure::TKEVD, args...) where FT = zero(FT)

@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::TKEVD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

@inline function viscous_flux_uz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::TKEVD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_vz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

@inline function viscous_flux_vz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::TKEVD, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_uz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...), # on boundaries, calculate fluxes explicitly
                  zero(FT))
end

