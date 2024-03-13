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

const VITD = VerticallyImplicitTimeDiscretization

#####
##### Terms in the turbulent kinetic energy equation, all at cell centers
#####

# Note special attention paid to averaging the vertical grid spacing correctly
@inline Δz_νₑ_∂z_u²ᶠᶜᶠ(i, j, k, grid, νₑ, u) = ℑxᶠᵃᵃ(i, j, k, grid, νₑ) * Δzᶠᶜᶠ(i, j, k, grid) * ∂zᶠᶜᶠ(i, j, k, grid, u)^2
@inline Δz_νₑ_∂z_v²ᶜᶠᶠ(i, j, k, grid, νₑ, v) = ℑyᵃᶠᵃ(i, j, k, grid, νₑ) * Δzᶜᶠᶠ(i, j, k, grid) * ∂zᶜᶠᶠ(i, j, k, grid, v)^2

@inline νₑ_∂z_u²ᶠᶜᶜ(i, j, k, grid, νₑ, u) = ℑzᵃᵃᶜ(i, j, k, grid, Δz_νₑ_∂z_u²ᶠᶜᶠ, νₑ, u) / Δzᶠᶜᶜ(i, j, k, grid) 
@inline νₑ_∂z_v²ᶜᶠᶜ(i, j, k, grid, νₑ, v) = ℑzᵃᵃᶜ(i, j, k, grid, Δz_νₑ_∂z_v²ᶜᶠᶠ, νₑ, v) / Δzᶜᶠᶜ(i, j, k, grid) 

@inline function shear_production(i, j, k, grid, νₑ, u, v)
    # Reconstruct the shear production term in an "approximately conservative" manner
    # (ie respecting the spatial discretization and using a stencil commensurate with the
    # loss of mean kinetic energy due to shear production --- but _not_ respecting the 
    # the temporal discretization. Note that also respecting the temporal discretization, would
    # require storing the velocity field at n and n+1):

    return ℑxᶜᵃᵃ(i, j, k, grid, νₑ_∂z_u²ᶠᶜᶜ, νₑ, u) +
           ℑyᵃᶜᵃ(i, j, k, grid, νₑ_∂z_v²ᶜᶠᶜ, νₑ, v)
end

# To reconstruct buoyancy flux "conservatively" (ie approximately correpsonding to production/destruction
# of mean potential energy):
@inline function buoyancy_fluxᶜᶜᶠ(i, j, k, grid, tracers, buoyancy, tracer_diffusivity)
    κᶜ = @inbounds tracer_diffusivity[i, j, k]
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return - κᶜ * N²
end

@inline explicit_buoyancy_flux(i, j, k, grid, tracers, buoyancy, κᶜ) =
    ℑzᵃᵃᶜ(i, j, k, grid, buoyancy_fluxᶜᶜᶠ, tracers, buoyancy, κᶜ)

@inline function turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
    eᵢ = @inbounds e[i, j, k]
    eᵐⁱⁿ = closure.minimum_turbulent_kinetic_energy
    return sqrt(max(eᵐⁱⁿ, eᵢ))
end

#####
##### Utilities for model constructors
#####

""" Infer tracer boundary conditions from user_bcs and tracer_names. """
function top_tracer_boundary_conditions(grid, tracer_names, user_bcs)
    default_tracer_bcs = NamedTuple(c => FieldBoundaryConditions(grid, (Center, Center, Center)) for c in tracer_names)
    bcs = merge(default_tracer_bcs, user_bcs)
    return NamedTuple(c => bcs[c].top for c in tracer_names)
end

""" Infer velocity boundary conditions from `user_bcs` and `tracer_names`. """
function top_velocity_boundary_conditions(grid, user_bcs)
    default_top_bc = default_prognostic_bc(topology(grid, 3)(), Center(), DefaultBoundaryCondition())

    user_bc_names = keys(user_bcs)
    u_top_bc = :u ∈ user_bc_names ? user_bcs.u.top : default_top_bc
    v_top_bc = :v ∈ user_bc_names ? user_bcs.v.top : default_top_bc

    return (u=u_top_bc, v=v_top_bc)
end

#####
##### Richardson number
#####

@inline Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy) =
    ℑzᵃᵃᶜ(i, j, k, grid, Riᶜᶜᶠ, velocities, tracers, buoyancy)

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    S² = ∂z_u² + ∂z_v²
    Ri = N² / S²
    return ifelse(N² ≤ 0, zero(grid), Ri)
end

const c = Center()
const f = Face()

@inline clip(x) = max(zero(x), x)

include("catke_vertical_diffusivity.jl")
include("catke_mixing_length.jl")
include("catke_equation.jl")
include("compute_catke_diffusivity_fields.jl")

include("tke_dissipation_vertical_diffusivity.jl")
include("tke_dissipation_equations.jl")
include("compute_tke_dissipation_diffusivity_fields.jl")

@inline viscosity_location(::FlavorOfCATKE)      = (c, c, f)
@inline diffusivity_location(::FlavorOfCATKE)    = (c, c, f)
@inline viscosity_location(::FlavorOfKEpsilon)   = (c, c, f)
@inline diffusivity_location(::FlavorOfKEpsilon) = (c, c, f)

for S in (:CATKEMixingLength, :CATKEEquation, :TKEDissipationEquations)
    @eval @inline convert_eltype(::Type{FT}, s::$S) where FT = $S{FT}(; Dict(p => getproperty(s, p) for p in propertynames(s))...)
    @eval @inline convert_eltype(::Type{FT}, s::$S{FT}) where FT = s
end

# For tuples of closures, we need to know _which_ closure is CATKE.
# Here we take a "simple" approach that sorts the tuple so CATKE is first.
# This is not sustainable though if multiple closures require this.
# The two other possibilities are:
# 1. Recursion to find which closure is CATKE in a compiler-inferrable way
# 2. Store the "CATKE index" inside CATKE via validate_closure.
validate_closure(closure_tuple::Tuple) = Tuple(sort(collect(closure_tuple), lt=catke_first))

catke_first(closure1, catke::FlavorOfCATKE) = false
catke_first(catke::FlavorOfCATKE, closure2) = true
catke_first(closure1, closure2) = false
catke_first(catke1::FlavorOfCATKE, catke2::FlavorOfCATKE) = error("Can't have two CATKEs in one closure tuple.")

end # module

