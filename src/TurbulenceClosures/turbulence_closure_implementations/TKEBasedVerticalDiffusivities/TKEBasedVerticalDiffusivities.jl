module TKEBasedVerticalDiffusivities

using Adapt
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

@inline Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy) =
    ℑbzᵃᵃᶜ(i, j, k, grid, Riᶜᶜᶠ, velocities, tracers, buoyancy)

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    S² = ∂z_u² + ∂z_v²
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

include("catke_vertical_diffusivity.jl")
include("catke_mixing_length.jl")
include("catke_equation.jl")
include("time_step_catke.jl")

for S in (:CATKEMixingLength, :CATKEEquation)
    @eval @inline convert_eltype(::Type{FT}, s::$S) where FT =
        $S{FT}(; Dict(p => getproperty(s, p) for p in propertynames(s))...)
    @eval @inline convert_eltype(::Type{FT}, s::$S{FT}) where FT = s
end

end # module

