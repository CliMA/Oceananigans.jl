using Oceananigans.Grids: LatitudeLongitudeGrid, OrthogonalSphericalShellGrid, φnode, hack_sind, hack_cosd, peripheral_node
using Oceananigans.Operators: Δz_qᶠᶜᶜ, Δz_qᶜᶠᶜ, Az_qᶜᶜᶠ, Δz⁻¹ᶜᶜᶠ, Az⁻¹ᶠᶜᶜ, Az⁻¹ᶜᶠᶜ,
                              ∂xᶠᶜᶜ, ∂yᶠᶜᶜ, ∂xᶜᶠᶜ, ∂yᶜᶠᶜ

"""
    HydrostaticFormulation

Tag indicating that the Coriolis force uses only the locally vertical component
of the rotation vector (the "traditional" approximation).
"""
struct HydrostaticFormulation end

"""
    NonhydrostaticFormulation

Tag indicating that the Coriolis force uses both the locally vertical and
horizontal components of the rotation vector.
"""
struct NonhydrostaticFormulation end

Base.summary(::HydrostaticFormulation) = "HydrostaticFormulation"
Base.summary(::NonhydrostaticFormulation) = "NonhydrostaticFormulation"

"""
    struct SphericalCoriolis{S, FT, F} <: AbstractRotation{S}

A Coriolis implementation for a sphere with latitude-dependent Coriolis parameter
`f = 2Ω sin(φ)`.
"""
struct SphericalCoriolis{S, FT, F} <: AbstractRotation{S}
    rotation_rate :: FT
    scheme :: S
    formulation :: F
end

const NonhydrostaticSchemes = Union{EnstrophyConserving, EnergyConserving, TriadScheme,
                                    ActiveWeightedEnstrophyConserving, ActiveWeightedEnergyConserving}

"""
    SphericalCoriolis([FT = Float64;]
                      rotation_rate = Oceananigans.defaults.planet_rotation_rate,
                      scheme = EnstrophyConserving(),
                      formulation = NonhydrostaticFormulation())

Return a parameter object for Coriolis forces on a sphere rotating at `rotation_rate`.

With the `NonhydrostaticFormulation` the Coriolis acceleration is the full `2Ω × U`: the traditional
part `f ẑ × U` with `f = 2Ω sin φ`, discretized with the horizontal stencil of `scheme`, plus the
nontraditional terms `(f̃ʸ w, -f̃ˣ w, f̃ˣ v - f̃ʸ u)` of the horizontal rotation vector `2Ω cos φ n̂`,
which are the same for every scheme and are projected onto the grid's axes on rotated grids.
With the `HydrostaticFormulation` only the traditional part is included.

Keyword arguments
=================

- `rotation_rate`: Sphere's rotation rate; default: Earth's rotation rate.
- `scheme`: Discretization scheme. Options include `EnstrophyConserving()` (default),
  `EnergyConserving()`, `ActiveWeightedEnstrophyConserving()`,
  `ActiveWeightedEnergyConserving()`, and `TriadScheme()`. With the `NonhydrostaticFormulation`
  these are the only options; `DualGridScheme` is supported by the `HydrostaticFormulation` only.
- `formulation`: Either `NonhydrostaticFormulation()` (default) or `HydrostaticFormulation()`.

Example
=======

```jldoctest
julia> using Oceananigans

julia> SphericalCoriolis()
SphericalCoriolis
├─ rotation rate: 7.29e-05 s⁻¹ = 1.00 Ω_Earth
├─ formulation: NonhydrostaticFormulation
└─ scheme: EnstrophyConserving
```
"""
function SphericalCoriolis(FT::DataType = Oceananigans.defaults.FloatType;
                           rotation_rate = Oceananigans.defaults.planet_rotation_rate,
                           scheme = EnstrophyConserving(),
                           formulation = NonhydrostaticFormulation())
    rotation_rate = convert(FT, rotation_rate)
    validate_coriolis_scheme(formulation, scheme)
    return SphericalCoriolis(rotation_rate, scheme, formulation)
end

validate_coriolis_scheme(formulation, scheme) = nothing
validate_coriolis_scheme(::NonhydrostaticFormulation, ::NonhydrostaticSchemes) = nothing
validate_coriolis_scheme(::NonhydrostaticFormulation, scheme) =
    throw(ArgumentError("SphericalCoriolis with NonhydrostaticFormulation() does not support $(summary(scheme))."))

const HydrostaticSphericalCoriolis{S, FT}      = SphericalCoriolis{S, FT, <:HydrostaticFormulation} where {S, FT}
const NonhydrostaticSphericalCoriolis{S, FT}   = SphericalCoriolis{S, FT, <:NonhydrostaticFormulation} where {S, FT}

"""
    HydrostaticSphericalCoriolis([FT = Float64;]
                                 rotation_rate = Oceananigans.defaults.planet_rotation_rate,
                                 scheme = EnstrophyConserving())

Return a `SphericalCoriolis` with `HydrostaticFormulation`. This is a convenience
constructor that sets `formulation = HydrostaticFormulation()` and defaults to the
`EnstrophyConserving` scheme.

Example
=======

```jldoctest
julia> using Oceananigans

julia> HydrostaticSphericalCoriolis()
SphericalCoriolis
├─ rotation rate: 7.29e-05 s⁻¹ = 1.00 Ω_Earth
├─ formulation: HydrostaticFormulation
└─ scheme: EnstrophyConserving
```
"""
function HydrostaticSphericalCoriolis(FT::DataType = Oceananigans.defaults.FloatType;
                                      rotation_rate = Oceananigans.defaults.planet_rotation_rate,
                                      scheme = EnstrophyConserving())
    return SphericalCoriolis(FT; rotation_rate, scheme, formulation=HydrostaticFormulation())
end

@inline φᶠᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, face)
@inline φᶠᶠᵃ(i, j, k, grid::OrthogonalSphericalShellGrid) = φnode(i, j, grid, face, face)
@inline φᶠᶠᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶠᶠᵃ(i, j, k, grid.underlying_grid)

@inline φᶠᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, center)
@inline φᶠᶜᵃ(i, j, k, grid::OrthogonalSphericalShellGrid) = φnode(i, j, grid, face, center)
@inline φᶠᶜᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶠᶜᵃ(i, j, k, grid.underlying_grid)

@inline φᶜᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, face)
@inline φᶜᶠᵃ(i, j, k, grid::OrthogonalSphericalShellGrid) = φnode(i, j, grid, center, face)
@inline φᶜᶠᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶜᶠᵃ(i, j, k, grid.underlying_grid)

@inline φᶜᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, center)
@inline φᶜᶜᵃ(i, j, k, grid::OrthogonalSphericalShellGrid) = φnode(i, j, grid, center, center)
@inline φᶜᶜᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶜᶜᵃ(i, j, k, grid.underlying_grid)

@inline fᶠᶠᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_sind(φᶠᶠᵃ(i, j, k, grid))
@inline fᶜᶜᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_sind(φᶜᶜᵃ(i, j, k, grid))
@inline f̃ᶠᶜᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_cosd(φᶠᶜᵃ(i, j, k, grid))
@inline f̃ᶜᶠᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_cosd(φᶜᶠᵃ(i, j, k, grid))

#####
##### The nonhydrostatic formulation adds the nontraditional terms (f̃ʸ w, -f̃ˣ w, f̃ˣ v - f̃ʸ u) of the horizontal
##### rotation vector f̃ n̂ = 2Ω cos φ n̂ to the traditional part, which every scheme discretizes with its own stencil.
#####

const OrthogonalSphericalShellGridOfSomeKind = Union{OrthogonalSphericalShellGrid,
                                                     ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:OrthogonalSphericalShellGrid}}

# The projections of north n̂ = a ∇φ onto the grid's axes, normalized so that they are exact on aligned grids
# n̂ ⋅ ŷ at (Face, Center)
@inline function northward_projection_yᶠᶜᵃ(i, j, k, grid)
    ∂x_φ = ∂xᶠᶜᶜ(i, j, k, grid, φᶜᶜᵃ)
    ∂y_φ = ∂yᶠᶜᶜ(i, j, k, grid, φᶠᶠᵃ)
    norm_∇φ = √(∂x_φ^2 + ∂y_φ^2)
    return ifelse(norm_∇φ == 0, zero(grid), ∂y_φ / norm_∇φ)
end

# n̂ ⋅ x̂ at (Center, Face)
@inline function northward_projection_xᶜᶠᵃ(i, j, k, grid)
    ∂φ∂x = ∂xᶜᶠᶜ(i, j, k, grid, φᶠᶠᵃ)
    ∂φ∂y = ∂yᶜᶠᶜ(i, j, k, grid, φᶜᶜᵃ)
    ∇φ = sqrt(∂φ∂x^2 + ∂φ∂y^2)
    return ifelse(∇φ == 0, zero(grid), ∂φ∂x / ∇φ)
end

# On a `LatitudeLongitudeGrid` y points north: f̃ʸ = f̃ and f̃ˣ = 0.
@inline f̃ʸᶠᶜᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = f̃ᶠᶜᵃ(i, j, k, grid, coriolis)

@inline f̃ʸᶠᶜᵃ(i, j, k, grid::OrthogonalSphericalShellGridOfSomeKind, coriolis::SphericalCoriolis) =
    f̃ᶠᶜᵃ(i, j, k, grid, coriolis) * northward_projection_yᶠᶜᵃ(i, j, k, grid)

@inline f̃ˣᶜᶠᵃ(i, j, k, grid::OrthogonalSphericalShellGridOfSomeKind, coriolis::SphericalCoriolis) =
    f̃ᶜᶠᵃ(i, j, k, grid, coriolis) * northward_projection_xᶜᶠᵃ(i, j, k, grid)

@inline f̃ʸ_ℑz_uᶠᶜᶠ(i, j, k, grid, coriolis, u) = f̃ʸᶠᶜᵃ(i, j, k, grid, coriolis) * ℑzᵃᵃᶠ(i, j, k, grid, Δz_qᶠᶜᶜ, u)
@inline f̃ʸ_ℑx_wᶠᶜᶠ(i, j, k, grid, coriolis, w) = f̃ʸᶠᶜᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w)
@inline f̃ˣ_ℑz_vᶜᶠᶠ(i, j, k, grid, coriolis, v) = f̃ˣᶜᶠᵃ(i, j, k, grid, coriolis) * ℑzᵃᵃᶠ(i, j, k, grid, Δz_qᶜᶠᶜ, v)
@inline f̃ˣ_ℑy_wᶜᶠᶠ(i, j, k, grid, coriolis, w) = f̃ˣᶜᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w)

# The nontraditional terms (f̃ʸ w, -f̃ˣ w, f̃ˣ v - f̃ʸ u). The f̃ˣ terms vanish where x̂ points east.
@inline x_f̃_cross_U(i, j, k, grid, coriolis, U) =
    @inbounds + ℑzᵃᵃᶜ(i, j, k, grid, f̃ʸ_ℑx_wᶠᶜᶠ, coriolis, U[3]) * Az⁻¹ᶠᶜᶜ(i, j, k, grid)

@inline y_f̃_cross_U(i, j, k, grid, coriolis, U) = zero(grid)
@inline y_f̃_cross_U(i, j, k, grid::OrthogonalSphericalShellGridOfSomeKind, coriolis, U) =
    @inbounds - ℑzᵃᵃᶜ(i, j, k, grid, f̃ˣ_ℑy_wᶜᶠᶠ, coriolis, U[3]) * Az⁻¹ᶜᶠᶜ(i, j, k, grid)

@inline z_f̃_cross_U(i, j, k, grid, coriolis, U) =
    @inbounds - ℑxᶜᵃᵃ(i, j, k, grid, f̃ʸ_ℑz_uᶠᶜᶠ, coriolis, U[1]) * Δz⁻¹ᶜᶜᶠ(i, j, k, grid)

@inline z_f̃_cross_U(i, j, k, grid::OrthogonalSphericalShellGridOfSomeKind, coriolis, U) =
    @inbounds - ℑxᶜᵃᵃ(i, j, k, grid, f̃ʸ_ℑz_uᶠᶜᶠ, coriolis, U[1]) * Δz⁻¹ᶜᶜᶠ(i, j, k, grid) +
                ℑyᵃᶜᵃ(i, j, k, grid, f̃ˣ_ℑz_vᶜᶠᶠ, coriolis, U[2]) * Δz⁻¹ᶜᶜᶠ(i, j, k, grid)

# Computes the traditional part with the scheme's stencil
@inline with_hydrostatic_formulation(coriolis::SphericalCoriolis) =
    SphericalCoriolis(coriolis.rotation_rate, coriolis.scheme, HydrostaticFormulation())

# One method per scheme avoids ambiguities with the `AbstractRotation{<:Scheme}` methods
for Scheme in (:EnstrophyConserving, :EnergyConserving, :TriadScheme,
               :ActiveWeightedEnstrophyConserving, :ActiveWeightedEnergyConserving)
    @eval begin
        @inline x_f_cross_U(i, j, k, grid, coriolis::NonhydrostaticSphericalCoriolis{<:$Scheme}, U) =
            x_f_cross_U(i, j, k, grid, with_hydrostatic_formulation(coriolis), U) + x_f̃_cross_U(i, j, k, grid, coriolis, U)

        @inline y_f_cross_U(i, j, k, grid, coriolis::NonhydrostaticSphericalCoriolis{<:$Scheme}, U) =
            y_f_cross_U(i, j, k, grid, with_hydrostatic_formulation(coriolis), U) + y_f̃_cross_U(i, j, k, grid, coriolis, U)

        @inline z_f_cross_U(i, j, k, grid, coriolis::NonhydrostaticSphericalCoriolis{<:$Scheme}, U) =
            z_f̃_cross_U(i, j, k, grid, coriolis, U)
    end
end

Adapt.adapt_structure(to, coriolis::SphericalCoriolis) =
    SphericalCoriolis(Adapt.adapt(to, coriolis.rotation_rate),
                      Adapt.adapt(to, coriolis.scheme),
                      Adapt.adapt(to, coriolis.formulation))

#####
##### Show
#####

function Base.show(io::IO, spherical_coriolis::SphericalCoriolis)
    coriolis_scheme = spherical_coriolis.scheme
    coriolis_formulation = spherical_coriolis.formulation
    rotation_rate   = spherical_coriolis.rotation_rate
    rotation_rate_Earth = Oceananigans.defaults.planet_rotation_rate
    rotation_rate_str = @sprintf("%.2e s⁻¹ = %.2f Ω_Earth", rotation_rate, rotation_rate / rotation_rate_Earth)

    return print(io, "SphericalCoriolis", '\n',
                 "├─ rotation rate: ", rotation_rate_str, '\n',
                 "├─ formulation: ", summary(coriolis_formulation), '\n',
                 "└─ scheme: ", summary(coriolis_scheme))
end
