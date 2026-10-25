using Oceananigans.Grids: LatitudeLongitudeGrid, OrthogonalSphericalShellGrid, AnySphericalShellGrid, φnode, hack_sind, hack_cosd
using Oceananigans.Operators: Δz_qᶠᶜᶜ, Δz_qᶜᶠᶜ, Az_qᶜᶜᶠ, Δz⁻¹ᶜᶜᶠ, Az⁻¹ᶠᶜᶜ, Az⁻¹ᶜᶠᶜ, rotation_angle

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

"""
    SphericalCoriolis([FT = Float64;]
                      rotation_rate = Oceananigans.defaults.planet_rotation_rate,
                      scheme = EnstrophyConserving(),
                      formulation = NonhydrostaticFormulation())

Return a parameter object for Coriolis forces on a sphere rotating at `rotation_rate`.

With the `NonhydrostaticFormulation` the Coriolis acceleration is the full `2Ω × U`: the traditional part
`f ẑ × U` with `f = 2Ω sin φ`, discretized with the horizontal stencil of `scheme`, plus the nontraditional
terms of the horizontal rotation vector `2Ω cos φ n̂`. With the `HydrostaticFormulation` only the traditional
part is included.

Keyword arguments
=================

- `rotation_rate`: Sphere's rotation rate; default: Earth's rotation rate.
- `scheme`: Discretization scheme. Options include `EnstrophyConserving()` (default),
  `EnergyConserving()`, `ActiveWeightedEnstrophyConserving()`,
  `ActiveWeightedEnergyConserving()`, and `TriadScheme()`.
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

    return SphericalCoriolis(rotation_rate, scheme, formulation)
end

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
@inline φᶠᶠᵃ(i, j, k, grid::AnySphericalShellGrid)        = φnode(i, j, grid, face, face)
@inline φᶠᶠᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶠᶠᵃ(i, j, k, grid.underlying_grid)

@inline φᶠᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, center)
@inline φᶠᶜᵃ(i, j, k, grid::AnySphericalShellGrid)        = φnode(i, j, grid, face, center)
@inline φᶠᶜᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶠᶜᵃ(i, j, k, grid.underlying_grid)

@inline φᶜᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, face)
@inline φᶜᶠᵃ(i, j, k, grid::AnySphericalShellGrid)        = φnode(i, j, grid, center, face)
@inline φᶜᶠᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶜᶠᵃ(i, j, k, grid.underlying_grid)

@inline φᶜᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, center)
@inline φᶜᶜᵃ(i, j, k, grid::AnySphericalShellGrid)        = φnode(i, j, grid, center, center)
@inline φᶜᶜᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶜᶜᵃ(i, j, k, grid.underlying_grid)

@inline fᶠᶠᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_sind(φᶠᶠᵃ(i, j, k, grid))
@inline fᶜᶜᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_sind(φᶜᶜᵃ(i, j, k, grid))
@inline f̃ᶠᶜᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_cosd(φᶠᶜᵃ(i, j, k, grid))
@inline f̃ᶜᶠᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_cosd(φᶜᶠᵃ(i, j, k, grid))

const OrthogonalSphericalShellGridOfSomeKind =
    Union{AnySphericalShellGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AnySphericalShellGrid}}

# The nontraditional terms (f̃ʸ w, -f̃ˣ w, f̃ˣ v - f̃ʸ u) of the horizontal rotation vector f̃ n̂ = 2Ω cos φ n̂, where
# n̂ points north, use its components (f̃ˣ, f̃ʸ) = f̃ (-sin θ, cos θ) along the grid's axes, with θ the angle
# from the grid's ŷ to north (`Operators.rotation_angle`), which vanishes on a `LatitudeLongitudeGrid`.
@inline north_angle(i, j, grid, ℓx, ℓy) = zero(grid)
@inline north_angle(i, j, grid::OrthogonalSphericalShellGridOfSomeKind, ℓx, ℓy) = rotation_angle(i, j, grid, ℓx, ℓy)

@inline f̃ʸᶠᶜᵃ(i, j, k, grid, coriolis) = + f̃ᶠᶜᵃ(i, j, k, grid, coriolis) * cos(north_angle(i, j, grid, face, center))
@inline f̃ˣᶜᶠᵃ(i, j, k, grid, coriolis) = - f̃ᶜᶠᵃ(i, j, k, grid, coriolis) * sin(north_angle(i, j, grid, center, face))

@inline f̃ʸ_ℑz_uᶠᶜᶠ(i, j, k, grid, coriolis, u) = f̃ʸᶠᶜᵃ(i, j, k, grid, coriolis) * ℑzᵃᵃᶠ(i, j, k, grid, Δz_qᶠᶜᶜ, u)
@inline f̃ʸ_ℑx_wᶠᶜᶠ(i, j, k, grid, coriolis, w) = f̃ʸᶠᶜᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w)
@inline f̃ˣ_ℑz_vᶜᶠᶠ(i, j, k, grid, coriolis, v) = f̃ˣᶜᶠᵃ(i, j, k, grid, coriolis) * ℑzᵃᵃᶠ(i, j, k, grid, Δz_qᶜᶠᶜ, v)
@inline f̃ˣ_ℑy_wᶜᶠᶠ(i, j, k, grid, coriolis, w) = f̃ˣᶜᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w)

@inline x_f̃_cross_U(i, j, k, grid, coriolis, U) =
    @inbounds + ℑzᵃᵃᶜ(i, j, k, grid, f̃ʸ_ℑx_wᶠᶜᶠ, coriolis, U[3]) * Az⁻¹ᶠᶜᶜ(i, j, k, grid)

@inline y_f̃_cross_U(i, j, k, grid, coriolis, U) =
    @inbounds - ℑzᵃᵃᶜ(i, j, k, grid, f̃ˣ_ℑy_wᶜᶠᶠ, coriolis, U[3]) * Az⁻¹ᶜᶠᶜ(i, j, k, grid)

@inline z_f̃_cross_U(i, j, k, grid, coriolis, U) =
    @inbounds (ℑyᵃᶜᵃ(i, j, k, grid, f̃ˣ_ℑz_vᶜᶠᶠ, coriolis, U[2]) -
               ℑxᶜᵃᵃ(i, j, k, grid, f̃ʸ_ℑz_uᶠᶜᶠ, coriolis, U[1])) * Δz⁻¹ᶜᶜᶠ(i, j, k, grid)

# The traditional part is the scheme's stencil, with one method per scheme to avoid ambiguities with the
# `AbstractRotation{<:Scheme}` methods
@inline hydrostatic(coriolis::SphericalCoriolis) =
    SphericalCoriolis(coriolis.rotation_rate, coriolis.scheme, HydrostaticFormulation())

for Scheme in (:EnstrophyConserving, :EnergyConserving, :TriadScheme,
               :ActiveWeightedEnstrophyConserving, :ActiveWeightedEnergyConserving, :DualGridScheme)
    @eval begin
        @inline x_f_cross_U(i, j, k, grid, coriolis::NonhydrostaticSphericalCoriolis{<:$Scheme}, U) =
            x_f_cross_U(i, j, k, grid, hydrostatic(coriolis), U) + x_f̃_cross_U(i, j, k, grid, coriolis, U)

        @inline y_f_cross_U(i, j, k, grid, coriolis::NonhydrostaticSphericalCoriolis{<:$Scheme}, U) =
            y_f_cross_U(i, j, k, grid, hydrostatic(coriolis), U) + y_f̃_cross_U(i, j, k, grid, coriolis, U)
    end
end

@inline z_f_cross_U(i, j, k, grid, coriolis::NonhydrostaticSphericalCoriolis, U) =
    z_f̃_cross_U(i, j, k, grid, coriolis, U)

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
