module TKEBasedVerticalDiffusivities

using Adapt
using CUDA
using KernelAbstractions: @kernel, @index

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Utils
using Oceananigans.Units
using Oceananigans.Fields
using Oceananigans.Operators

using Oceananigans.Utils: prettysummary
using Oceananigans.Grids: peripheral_node, inactive_node, inactive_cell
using Oceananigans.Fields: ZeroField
using Oceananigans.BoundaryConditions: default_prognostic_bc, DefaultBoundaryCondition
using Oceananigans.BoundaryConditions: BoundaryCondition, FieldBoundaryConditions
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction, FluxBoundaryCondition
using Oceananigans.BuoyancyModels: ∂z_b, top_buoyancy_flux
using Oceananigans.Grids: inactive_cell

using Oceananigans.TurbulenceClosures:
    getclosure,
    time_discretization,
    AbstractScalarDiffusivity,
    VerticallyImplicitTimeDiscretization,
    VerticalFormulation
    
import Oceananigans.BoundaryConditions: getbc
import Oceananigans.Utils: with_tracers
import Oceananigans.TurbulenceClosures:
    validate_closure,
    shear_production,
    buoyancy_flux,
    dissipation,
    add_closure_specific_boundary_conditions,
    compute_diffusivities!,
    DiffusivityFields,
    implicit_linear_coefficient,
    viscosity,
    diffusivity,
    viscosity_location,
    diffusivity_location,
    diffusive_flux_x,
    diffusive_flux_y,
    diffusive_flux_z

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

@inline Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy) =
    ℑbzᵃᵃᶜ(i, j, k, grid, Riᶜᶜᶠ, velocities, tracers, buoyancy)

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    u = velocities.u
    v = velocities.v
    S² = shearᶜᶜᶠ(i, j, k, grid, u, v)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    Ri = N² / S²
    return ifelse(N² == 0, zero(grid), Ri)
end

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
@inline function buoyancy_fluxᶜᶜᶠ(i, j, k, grid, tracers, buoyancy, diffusivities)
    κc = @inbounds diffusivities.κc[i, j, k]
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return - κc * N²
end
 
@inline explicit_buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivities) =
    ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_fluxᶜᶜᶠ, tracers, buoyancy, diffusivities)

# Note special attention paid to averaging the vertical grid spacing correctly
@inline Δz_νₑ_az_bzᶠᶜᶠ(i, j, k, grid, νₑ, a, b) = ℑxᶠᵃᵃ(i, j, k, grid, νₑ) * ∂zᶠᶜᶠ(i, j, k, grid, a) * 
                                                  Δzᶠᶜᶠ(i, j, k, grid)     * ∂zᶠᶜᶠ(i, j, k, grid, b)

@inline Δz_νₑ_az_bzᶜᶠᶠ(i, j, k, grid, νₑ, a, b) = ℑyᵃᶠᵃ(i, j, k, grid, νₑ) * ∂zᶜᶠᶠ(i, j, k, grid, a) * 
                                                  Δzᶜᶠᶠ(i, j, k, grid)     * ∂zᶜᶠᶠ(i, j, k, grid, b)

@inline function shear_production_xᶠᶜᶜ(i, j, k, grid, νₑ, uⁿ, u⁺)
    Δz_Pxⁿ = ℑbzᵃᵃᶜ(i, j, k, grid, Δz_νₑ_az_bzᶠᶜᶠ, νₑ, uⁿ, u⁺)
    Δz_Px⁺ = ℑbzᵃᵃᶜ(i, j, k, grid, Δz_νₑ_az_bzᶠᶜᶠ, νₑ, u⁺, u⁺)
    return (Δz_Pxⁿ + Δz_Px⁺) / (2 * Δzᶠᶜᶜ(i, j, k, grid))
end

@inline function shear_production_yᶜᶠᶜ(i, j, k, grid, νₑ, vⁿ, v⁺)
    Δz_Pyⁿ = ℑbzᵃᵃᶜ(i, j, k, grid, Δz_νₑ_az_bzᶜᶠᶠ, νₑ, vⁿ, v⁺)
    Δz_Py⁺ = ℑbzᵃᵃᶜ(i, j, k, grid, Δz_νₑ_az_bzᶜᶠᶠ, νₑ, v⁺, v⁺)
    return (Δz_Pyⁿ + Δz_Py⁺) / (2 * Δzᶜᶠᶜ(i, j, k, grid))
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
    closure = CUDA.@allowscalar closure_array[1, 1]
    return get_time_step(closure)
end

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

