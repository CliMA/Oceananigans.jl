using Oceananigans.Fields: AbstractField
using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶜᶜᶜ

const c = Center()
const f = Face()

""" Return the geopotential height at `i, j, k` at cell centers. """
@inline Zᶜᶜᶜ(i, j, k, grid) =
    ifelse(k < 1,       znode(i, j,       1, grid, c, c, c, with_halos=true) + (1 - k) * Δzᶜᶜᶠ(i, j, 1, grid),
    ifelse(k > grid.Nz, znode(i, j, grid.Nz, grid, c, c, c, with_halos=true) - (k - grid.Nz) * Δzᶜᶜᶠ(i, j, grid.Nz, grid),
                        znode(i, j,       k, grid, c, c, c, with_halos=true)))

""" Return the geopotential height at `i, j, k` at cell z-interfaces. """
@inline Zᶜᶜᶠ(i, j, k, grid) =
    ifelse(k < 1,           znode(i, j,           1, grid, c, c, f, with_halos=true) + (1 - k) * Δzᶜᶜᶜ(i, j, 1, grid),
    ifelse(k > grid.Nz + 1, znode(i, j, grid.Nz + 1, grid, c, c, f, with_halos=true) - (k - grid.Nz + 1) * Δzᶜᶜᶜ(i, j, k, grid),
                            znode(i, j,           k, grid, c, c, f, with_halos=true)))

# Dispatch shenanigans
@inline θ_and_sᴬ(i, j, k, θ::AbstractArray, sᴬ::AbstractArray) = @inbounds θ[i, j, k], sᴬ[i, j, k]
@inline θ_and_sᴬ(i, j, k, θ::Number,        sᴬ::AbstractArray) = @inbounds θ, sᴬ[i, j, k]
@inline θ_and_sᴬ(i, j, k, θ::AbstractArray, sᴬ::Number)        = @inbounds θ[i, j, k], sᴬ
@inline θ_and_sᴬ(i, j, k, θ::Number,        sᴬ::Number)        = @inbounds θ, sᴬ

# Basic functionality
@inline ρ′(i, j, k, grid, eos, θ, sᴬ) = @inbounds ρ′(θ_and_sᴬ(i, j, k, θ, sᴬ)..., Zᶜᶜᶜ(i, j, k, grid), eos)

@inline thermal_expansionᶜᶜᶜ(i, j, k, grid, eos, θ, sᴬ) = thermal_expansion(θ_and_sᴬ(i, j, k, θ, sᴬ)..., Zᶜᶜᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶠᶜᶜ(i, j, k, grid, eos, θ, sᴬ) = thermal_expansion(ℑxᶠᵃᵃ(i, j, k, grid, θ), ℑxᶠᵃᵃ(i, j, k, grid, sᴬ), Zᶜᶜᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶜᶠᶜ(i, j, k, grid, eos, θ, sᴬ) = thermal_expansion(ℑyᵃᶠᵃ(i, j, k, grid, θ), ℑyᵃᶠᵃ(i, j, k, grid, sᴬ), Zᶜᶜᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶜᶜᶠ(i, j, k, grid, eos, θ, sᴬ) = thermal_expansion(ℑzᵃᵃᶠ(i, j, k, grid, θ), ℑzᵃᵃᶠ(i, j, k, grid, sᴬ), Zᶜᶜᶠ(i, j, k, grid), eos)

@inline haline_contractionᶜᶜᶜ(i, j, k, grid, eos, θ, sᴬ) = haline_contraction(θ_and_sᴬ(i, j, k, θ, sᴬ)..., Zᶜᶜᶜ(i, j, k, grid), eos)
@inline haline_contractionᶠᶜᶜ(i, j, k, grid, eos, θ, sᴬ) = haline_contraction(ℑxᶠᵃᵃ(i, j, k, grid, θ), ℑxᶠᵃᵃ(i, j, k, grid, sᴬ), Zᶜᶜᶜ(i, j, k, grid), eos)
@inline haline_contractionᶜᶠᶜ(i, j, k, grid, eos, θ, sᴬ) = haline_contraction(ℑyᵃᶠᵃ(i, j, k, grid, θ), ℑyᵃᶠᵃ(i, j, k, grid, sᴬ), Zᶜᶜᶜ(i, j, k, grid), eos)
@inline haline_contractionᶜᶜᶠ(i, j, k, grid, eos, θ, sᴬ) = haline_contraction(ℑzᵃᵃᶠ(i, j, k, grid, θ), ℑzᵃᵃᶠ(i, j, k, grid, sᴬ), Zᶜᶜᶠ(i, j, k, grid), eos)
