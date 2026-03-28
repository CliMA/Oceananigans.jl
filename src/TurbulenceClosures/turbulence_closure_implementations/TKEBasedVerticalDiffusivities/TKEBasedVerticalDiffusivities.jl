module TKEBasedVerticalDiffusivities

export CATKEVerticalDiffusivity,
       TKEDissipationVerticalDiffusivity

using Adapt: Adapt, adapt
using GPUArraysCore: @allowscalar
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans
using Oceananigans.Grids: Center, Face, peripheral_node, inactive_node, inactive_cell, static_column_depthᶜᶜᵃ
using Oceananigans.Fields: CenterField, XFaceField, YFaceField, ZFaceField, ZeroField
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶠᶠ, Δzᶠᶜᶠ, Δz⁻¹ᶜᶠᶜ, Δz⁻¹ᶠᶜᶜ,
    ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ, ℑxyᶜᶜᵃ, ∂zᶜᶠᶠ, ∂zᶠᶜᶠ, ∂zᶠᶠᶠ
using Oceananigans.Utils: Utils, launch!, prettysummary

using Oceananigans.BoundaryConditions:
    BoundaryConditions,
    default_prognostic_bc,
    fill_halo_regions!,
    DefaultBoundaryCondition,
    FieldBoundaryConditions,
    DiscreteBoundaryFunction,
    FluxBoundaryCondition

using Oceananigans.BuoyancyFormulations:
    BuoyancyForce,
    BuoyancyTracer,
    SeawaterBuoyancy,
    TemperatureSeawaterBuoyancy,
    SalinitySeawaterBuoyancy,
    ∂z_b,
    ∂z_bᶠᶠᶠ,
    top_buoyancy_flux

using Oceananigans.TurbulenceClosures:
    getclosure,
    time_discretization,
    AbstractScalarDiffusivity,
    VerticallyImplicitTimeDiscretization,
    VerticalFormulation

import Oceananigans: prognostic_state, restore_prognostic_state!

import Oceananigans.TurbulenceClosures:
    validate_closure,
    shear_production,
    dissipation,
    buoyancy_force,
    buoyancy_tracers,
    add_closure_specific_boundary_conditions,
    closure_required_tracers,
    compute_closure_fields!,
    step_closure_prognostics!,
    build_closure_fields,
    implicit_linear_coefficient,
    viscosity,
    diffusivity,
    viscosity_location,
    diffusivity_location

const c = Center()
const f = Face()
const VITD = VerticallyImplicitTimeDiscretization

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function shearᶜᶜᶠ(i, j, k, grid, u, v)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, v)
    S² = ∂z_u² + ∂z_v²
    return S²
end

@inline function shearᶜᶜᶜ(i, j, k, grid, u, v)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ℑbzᵃᵃᶜ, ϕ², ∂zᶠᶜᶠ, u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ℑbzᵃᵃᶜ, ϕ², ∂zᶜᶠᶠ, v)
    S² = ∂z_u² + ∂z_v²
    return S²
end

@inline function shear_squaredᶠᶠᶠ(i, j, k, grid, velocities)
    ∂z_u² = ∂zᶠᶠᶠ(i, j, k, grid, ℑyᵃᶠᵃ, velocities.u)^2
    ∂z_v² = ∂zᶠᶠᶠ(i, j, k, grid, ℑxᶠᵃᵃ, velocities.v)^2
    return ∂z_u² + ∂z_v²
end

@inline function Riᶠᶠᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    S² = shear_squaredᶠᶠᶠ(i, j, k, grid, velocities)
    N² = ∂z_bᶠᶠᶠ(i, j, k, grid, buoyancy, tracers)
    Ri = N² / S²
    return ifelse(N² <= 0, zero(grid), Ri)
end

@inline Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy) =
    ℑxyᶜᶜᵃ(i, j, k, grid, Riᶠᶠᶠ, velocities, tracers, buoyancy)

@inline Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy) =
    ℑbzᵃᵃᶜ(i, j, k, grid, Riᶜᶜᶠ, velocities, tracers, buoyancy)

# @inline ℑbzᵃᵃᶜ(i, j, k, grid, fᵃᵃᶠ, args...) = ℑzᵃᵃᶜ(i, j, k, grid, fᵃᵃᶠ, args...)

# A particular kind of reconstruction that ignores peripheral nodes
@inline function ℑbzᵃᵃᶜ(i, j, k, grid, fᵃᵃᶠ, args...)
    k⁺ = k + 1
    k⁻ = k

    f⁺ = fᵃᵃᶠ(i, j, k⁺, grid, args...)
    f⁻ = fᵃᵃᶠ(i, j, k⁻, grid, args...)

    p⁺ = peripheral_node(i, j, k⁺, grid, c, c, f)
    p⁻ = peripheral_node(i, j, k⁻, grid, c, c, f)

    f⁺ = ifelse(p⁺, f⁻, f⁺)
    f⁻ = ifelse(p⁻, f⁺, f⁻)

    return (f⁺ + f⁻) / 2
end

# To reconstruct buoyancy flux "conservatively" (ie approximately corresponding to production/destruction
# of mean potential energy):
@inline function buoyancy_fluxᶜᶜᶠ(i, j, k, grid, tracers, buoyancy, closure_fields)
    κc = @inbounds closure_fields.κc[i, j, k]
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return - κc * N²
end

@inline explicit_buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, closure_fields) =
    ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_fluxᶜᶜᶠ, tracers, buoyancy, closure_fields)

# Note special attention paid to averaging the vertical grid spacing correctly
@inline Δz_νₑ_az_bzᶠᶜᶠ(i, j, k, grid, νₑ, a, b) = ℑxᶠᵃᵃ(i, j, k, grid, νₑ) * ∂zᶠᶜᶠ(i, j, k, grid, a) *
                                                  Δzᶠᶜᶠ(i, j, k, grid)     * ∂zᶠᶜᶠ(i, j, k, grid, b)

@inline Δz_νₑ_az_bzᶜᶠᶠ(i, j, k, grid, νₑ, a, b) = ℑyᵃᶠᵃ(i, j, k, grid, νₑ) * ∂zᶜᶠᶠ(i, j, k, grid, a) *
                                                  Δzᶜᶠᶠ(i, j, k, grid)     * ∂zᶜᶠᶠ(i, j, k, grid, b)

@inline function shear_production_xᶠᶜᶜ(i, j, k, grid, νₑ, uⁿ, u⁺)
    Δz_Pxⁿ = ℑbzᵃᵃᶜ(i, j, k, grid, Δz_νₑ_az_bzᶠᶜᶠ, νₑ, uⁿ, u⁺)
    Δz_Px⁺ = ℑbzᵃᵃᶜ(i, j, k, grid, Δz_νₑ_az_bzᶠᶜᶠ, νₑ, u⁺, u⁺)
    return (Δz_Pxⁿ + Δz_Px⁺) / 2 * Δz⁻¹ᶠᶜᶜ(i, j, k, grid)
end

@inline function shear_production_yᶜᶠᶜ(i, j, k, grid, νₑ, vⁿ, v⁺)
    Δz_Pyⁿ = ℑbzᵃᵃᶜ(i, j, k, grid, Δz_νₑ_az_bzᶜᶠᶠ, νₑ, vⁿ, v⁺)
    Δz_Py⁺ = ℑbzᵃᵃᶜ(i, j, k, grid, Δz_νₑ_az_bzᶜᶠᶠ, νₑ, v⁺, v⁺)
    return (Δz_Pyⁿ + Δz_Py⁺) / 2 * Δz⁻¹ᶜᶠᶜ(i, j, k, grid)
end

@inline function shear_production(i, j, k, grid, νₑ, uⁿ, u⁺, vⁿ, v⁺)
    # Reconstruct the shear production term in an "approximately conservative" manner
    # (ie respecting the spatial discretization and using a stencil commensurate with the
    # loss of mean kinetic energy due to shear production --- but _not_ respecting the
    # the temporal discretization. Note that also respecting the temporal discretization, would
    # require storing the velocity field at n and n+1):

    return ℑxᶜᵃᵃ(i, j, k, grid, shear_production_xᶠᶜᶜ, νₑ, uⁿ, u⁺) +
           ℑyᵃᶜᵃ(i, j, k, grid, shear_production_yᶜᶠᶜ, νₑ, vⁿ, v⁺)
end

@inline function turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
    eᵢ = @inbounds e[i, j, k]
    eᵐⁱⁿ = closure.minimum_tke
    return sqrt(max(eᵐⁱⁿ, eᵢ))
end

@inline function mask_diffusivity(i, j, k, grid, κ★)
    on_periphery = peripheral_node(i, j, k, grid, c, c, f)
    within_inactive = inactive_node(i, j, k, grid, c, c, f)
    nan = convert(eltype(grid), NaN)
    return ifelse(on_periphery, zero(grid), ifelse(within_inactive, nan, κ★))
end

@inline clip(x) = max(zero(x), x)

function get_time_step(closure_array::AbstractArray)
    # assume they are all the same
    closure = @allowscalar closure_array[1, 1]
    return get_time_step(closure)
end

get_top_tracer_bcs(bf::BuoyancyForce, tracers) = get_top_tracer_bcs(bf.formulation, tracers)
get_top_tracer_bcs(::Nothing, tracers) = NamedTuple()
get_top_tracer_bcs(::BuoyancyTracer, tracers) = (; b=tracers.b.boundary_conditions.top)
get_top_tracer_bcs(::SeawaterBuoyancy, tracers) = (T = tracers.T.boundary_conditions.top,
                                                   S = tracers.S.boundary_conditions.top)
get_top_tracer_bcs(::TemperatureSeawaterBuoyancy, tracers) = (; T = tracers.T.boundary_conditions.top)
get_top_tracer_bcs(::SalinitySeawaterBuoyancy, tracers)    = (; S = tracers.S.boundary_conditions.top)

include("tke_top_boundary_condition.jl")
include("catke_vertical_diffusivity.jl")
include("catke_mixing_length.jl")
include("catke_equation.jl")
include("time_step_catke_equation.jl")

include("tke_dissipation_vertical_diffusivity.jl")
include("tke_dissipation_stability_functions.jl")
include("tke_dissipation_equations.jl")

for S in (:CATKEMixingLength,
          :CATKEEquation,
          :StratifiedDisplacementScale,
          :ConstantStabilityFunctions,
          :VariableStabilityFunctions)

    @eval @inline convert_eltype(::Type{FT}, s::$S) where FT =
        $S{FT}(; Dict(p => getproperty(s, p) for p in propertynames(s))...)

    @eval @inline convert_eltype(::Type{FT}, s::$S{FT}) where FT = s
end

end # module
