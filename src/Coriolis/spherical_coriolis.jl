using Oceananigans.Grids: LatitudeLongitudeGrid, OrthogonalSphericalShellGrid, φnode, hack_sind, hack_cosd, peripheral_node
using Oceananigans.Operators
using Oceananigans.Operators: Δx_qᶜᶠᶜ, Δy_qᶠᶜᶜ, Δz_qᶠᶜᶜ, Δx_qᶜᶜᶠ, Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, Δz⁻¹ᶜᶜᶠ, ℑxyᶠᶜᵃ, ℑxyᶠᶜᵃ

struct HydrostaticFormulation end
struct NonhydrostaticFormulation end

struct SphericalCoriolis{S, FT, F} <: AbstractRotation{S}
    rotation_rate :: FT
    scheme :: S
    formulation :: F
end

"""
    SphericalCoriolis([FT=Float64;]
                      rotation_rate = Ω_Earth,
                      scheme = EnstrophyConserving())
                      formulation = HydrostaticFormulation()

Return a parameter object for Coriolis forces on a sphere rotating at `rotation_rate`.

Keyword arguments
=================

- `rotation_rate`: Sphere's rotation rate; default: [`Ω_Earth`](@ref).
- `scheme`: Either `EnergyConserving()` or `EnstrophyConserving()` (default).
- `formulation`: `HydrostaticFormulation` or `NonhydrostaticFormulation`.
"""
function SphericalCoriolis(FT::DataType = Oceananigans.defaults.FloatType;
                           rotation_rate = Oceananigans.defaults.planet_rotation_rate,
                           scheme = EnstrophyConserving(FT),
                           formulation = NonhydrostaticFormulation())
    rotation_rate = convert(FT, rotation_rate)

    return SphericalCoriolis(rotation_rate, scheme, formulation)
end

const HydrostaticSphericalCoriolis{S, FT}     = SphericalCoriolis{S, FT, <:HydrostaticFormulation} where {S, FT}
const NonhystrostaticSphericalCoriolis{S, FT} = SphericalCoriolis{S, FT, <:NonhydrostaticFormulation} where {S, FT}

function HydrostaticSphericalCoriolis(FT::DataType = Oceananigans.defaults.FloatType;
                                      rotation_rate = Oceananigans.defaults.planet_rotation_rate,
                                      scheme = EENConserving())
    return SphericalCoriolis(FT; rotation_rate, scheme, formulation=HydrostaticFormulation())
end

@inline φᶠᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, face)
@inline φᶠᶠᵃ(i, j, k, grid::OrthogonalSphericalShellGrid) = φnode(i, j, grid, face, face)
@inline φᶠᶠᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶠᶠᵃ(i, j, k, grid.underlying_grid)

@inline φᶠᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, center)
@inline φᶠᶜᵃ(i, j, k, grid::OrthogonalSphericalShellGrid) = φnode(i, j, grid, face, center)
@inline φᶠᶜᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶠᶜᵃ(i, j, k, grid.underlying_grid)

@inline φᶜᶜᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, center)
@inline φᶜᶜᵃ(i, j, k, grid::OrthogonalSphericalShellGrid) = φnode(i, j, grid, center, center)
@inline φᶜᶜᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶜᶜᵃ(i, j, k, grid.underlying_grid)

@inline fᶠᶠᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_sind(φᶠᶠᵃ(i, j, k, grid))
@inline fᶜᶜᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_sind(φᶜᶜᵃ(i, j, k, grid))
@inline f̃ᶠᶜᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_cosd(φᶠᶜᵃ(i, j, k, grid))

@inline f̃_ℑz_uᶠᶜᶠ(i, j, k, grid, coriolis::NonhystrostaticSphericalCoriolis, u) = f̃ᶠᶜᵃ(i, j, k, grid, coriolis) * ℑzᵃᵃᶠ(i, j, k, grid, Δz_qᶠᶜᶜ, u)
@inline f̃_ℑx_wᶠᶜᶠ(i, j, k, grid, coriolis::NonhystrostaticSphericalCoriolis, w) = f̃ᶠᶜᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶜᶠ, w)

const ENNC = NonhystrostaticSphericalCoriolis{<:EnstrophyConserving}

@inline x_f_cross_U(i, j, k, grid, coriolis::ENNC, U) = @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_vᶠᶠᶜ, coriolis, U[2]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid) +
                                                                    ℑzᵃᵃᶜ(i, j, k, grid, f̃_ℑx_wᶠᶜᶠ, coriolis, U[3]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, coriolis::ENNC, U) = @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_uᶠᶠᶜ, coriolis, U[1]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
@inline z_f_cross_U(i, j, k, grid, coriolis::ENNC, U) = @inbounds - ℑxᶜᵃᵃ(i, j, k, grid, f̃_ℑz_uᶠᶜᶠ, coriolis, U[1]) * Δz⁻¹ᶜᶜᶠ(i, j, k, grid)

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
    rotation_rate_str = @sprintf("%.2e s⁻¹ = %.2e Ω_Earth", rotation_rate, rotation_rate_Earth)

    return print(io, "SphericalCoriolis", '\n',
                 "├─ rotation rate: ", rotation_rate_str, '\n',
                 "├─ formulation: ", summary(coriolis_formulation), '\n',
                 "└─ scheme: ", summary(coriolis_scheme))
end
