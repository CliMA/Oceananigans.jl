using Oceananigans.Grids: LatitudeLongitudeGrid, OrthogonalSphericalShellGrid, peripheral_node, φnode
using Oceananigans.Operators: Δx_qᶜᶠᶜ, Δy_qᶠᶜᶜ, Δxᶠᶜᶜ, Δyᶜᶠᶜ, hack_sind, hack_cosd
using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.ImmersedBoundaries


"""
Will use the following syntax:

hydrostatic_coriolis = HydrostaticSphericalCoriolis()
full_coriolis = SphericalCoriolis()

"""

struct HydrostaticFormulation end
struct NonhydrostaticFormulation end


"""
    struct SphericalCoriolis{S, FT, F} 

A parameter object for constant rotation around a vertical axis on the sphere.
"""

struct SphericalCoriolis{S, FT, F}
    rotation_rate :: FT
    scheme :: S
    formulation :: F
end

"""
    SphericalCoriolis([FT=Float64;]
                                 rotation_rate = Ω_Earth,
                                 scheme = EnstrophyConserving())
                                 formulation = HydrostaticFormulation/NonhydrostaticFormulation 

Return a parameter object for Coriolis forces on a sphere rotating at `rotation_rate`.

Keyword arguments
=================

- `rotation_rate`: Sphere's rotation rate; default: [`Ω_Earth`](@ref).
- `scheme`: Either `EnergyConserving()`, `EnstrophyConserving()`, or `EnstrophyConserving()` (default).
- 'formulation`: `HydrostaticFormulation` or `NonhydrostaticFormulation`.
"""

const HydrostaticSphericalCoriolis{S, FT} = SphericalCoriolis{S, FT, <:HydrostaticFormulation} where {S, FT}

function SphericalCoriolis(FT::DataType=Oceananigans.defaults.FloatType;
                           rotation_rate = Ω_Earth,
                           scheme = EnstrophyConserving(FT),
                           formulation = HydrostaticFormulation()) 

    return SphericalCoriolis(rotation_rate, scheme, formulation)
end

Adapt.adapt_structure(to, coriolis::SphericalCoriolis) =
    SphericalCoriolis(Adapt.adapt(to, coriolis.rotation_rate),
                      Adapt.adapt(to, coriolis.scheme),
                      Adapt.adapt(to, coriolis.formulation))

@inline φᶠᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, face)
@inline φᶠᶠᵃ(i, j, k, grid::OrthogonalSphericalShellGrid) = φnode(i, j, grid, face, face)
@inline φᶠᶠᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶠᶠᵃ(i, j, k, grid.underlying_grid)

@inline fᶠᶠᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_sind(φᶠᶠᵃ(i, j, k, grid))
@inline f̃ᶠᶠᵃ(i, j, k, grid, coriolis::SphericalCoriolis) = 2 * coriolis.rotation_rate * hack_cosd(φᶠᶠᵃ(i, j, k, grid)

@inline f_ℑy_uᶠᶠᵃ(i, j, k, grid, coriolis, u) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)
@inline f_ℑx_vᶠᶠᵃ(i, j, k, grid, coriolis, v) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)

@inline f̃_ℑz_uᶠᶠᵃ(i, j, k, grid, coriolis, u) = f̃ᶠᶠᵃ(i, j, k, grid, coriolis) * ℑzᵃᶠᵃ(i, j, k, grid, Δz_qᶜᶜᶠ, u)
@inline f̃_ℑx_wᶠᶠᵃ(i, j, k, grid, coriolis, w) = f̃ᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, w)

@inline f̃_ℑz_uᶠᶠᵃ(i, j, k, grid, ::HydrostaticSphericalCoriolis, u) = zero(grid)
@inline f̃_ℑx_wᶠᶠᵃ(i, j, k, grid, ::HydrostaticSphericalCoriolis, w) = zero(grid)

@inline x_f_cross_U(i, j, k, grid, U) = @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_vᶠᶠᵃ, U[2]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid) + ℑzᵃᶜᵃ(i, j, k, grid, f̃_ℑx_wᶠᶠᵃ, U[3]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline y_f_cross_U(i, j, k, grid, U) = @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_uᶠᶠᵃ, U[1]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
@inline z_f_cross_U(i, j, k, grid, U) = @inbounds - ℑxᶜᵃᵃ(i, j, k, grid, f̃_ℑz_uᶠᶠᵃ, U[1]) * Δz⁻¹ᶜᶜᶠ(i, j, k, grid)

@inline z_f_cross_U(i, j, k, grid, ::HydrostaticSphericalCoriolis, U) = zero(grid)

#####
##### Active Point Enstrophy-conserving scheme
#####

# It might happen that a cell is active but all the neighbouring staggered nodes are inactive,
# (an example is a 1-cell large channel)
# In that case the Coriolis force is equal to zero

const CoriolisEnstrophyConserving = SphericalCoriolis{<:HydrostaticFormulation, <:EnstrophyConserving}

@inline x_f_cross_U(i, j, k, grid, coriolis::CoriolisEnstrophyConserving, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) *
                active_weighted_ℑxyᶠᶜᶜ(i, j, k, grid, Δx_qᶜᶠᶜ, U[2]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, coriolis::CoriolisEnstrophyConserving, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) *
                active_weighted_ℑxyᶜᶠᶜ(i, j, k, grid, Δy_qᶠᶜᶜ, U[1]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)

#####
##### Energy-conserving scheme
#####

const CoriolisEnergyConserving = SphericalCoriolis{<:HydrostaticFormulation, <:EnergyConserving}

@inline f_ℑx_vᶠᶠᵃ(i, j, k, grid, coriolis, v) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)
@inline f_ℑy_uᶠᶠᵃ(i, j, k, grid, coriolis, u) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)

@inline x_f_cross_U(i, j, k, grid, coriolis::CoriolisEnergyConserving, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_vᶠᶠᵃ, coriolis, U[2]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, coriolis::CoriolisEnergyConserving, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_uᶠᶠᵃ, coriolis, U[1]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)


#####
##### Show
#####

function Base.show(io::IO, spherical_coriolis::SphericalCoriolis)
    coriolis_scheme = spherical_coriolis.scheme
    coriolis_formulation = spherical_coriolis.formulation
    rotation_rate   = spherical_coriolis.rotation_rate
    rotation_rate_Earth = rotation_rate / Ω_Earth
    rotation_rate_str = @sprintf("%.2e s⁻¹ = %.2e Ω_Earth", rotation_rate, rotation_rate_Earth)

    return print(io, "SphericalCoriolis", '\n',
                 "├─ rotation rate: ", rotation_rate_str, '\n',
                 "├─ formulation: ", summary(coriolis_formulation), '\n',
                 "└─ scheme: ", summary(coriolis_scheme))
end
