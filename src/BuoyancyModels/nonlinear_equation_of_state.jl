using Oceananigans.Fields: AbstractField
using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᵃᵃᶠ, Δzᵃᵃᶜ

""" Return the geopotential height at `i, j, k` at cell centers. """
@inline function Zᵃᵃᶜ(i, j, k, grid::AbstractGrid{FT}) where FT
    if k < 1
        return znode(Center(), 1, grid) + (1 - k) * Δzᵃᵃᶠ(i, j, 1, grid)
    elseif k > grid.Nz
        return znode(Center(), grid.Nz, grid) - (k - grid.Nz) * Δzᵃᵃᶠ(i, j, grid.Nz, grid)
    else
        return znode(Center(), k, grid)
    end
end

""" Return the geopotential height at `i, j, k` at cell z-interfaces. """
@inline function Zᵃᵃᶠ(i, j, k, grid::AbstractGrid{FT}) where FT
    if k < 1
        return znode(Face(), 1, grid) + (1 - k) * Δzᵃᵃᶜ(i, j, 1, grid)
    elseif k > grid.Nz + 1
        return znode(Face(), grid.Nz + 1, grid) - (k - grid.Nz + 1) * Δzᵃᵃᶜ(i, j, k, grid)
    else
        return znode(Face(), k, grid)
    end
end

const ArrayOrField = Union{AbstractArray, AbstractField}

# Dispatch shenanigans
@inline θ_and_sᴬ(i, j, k, θ::ArrayOrField, sᴬ::ArrayOrField) = @inbounds θ[i, j, k], sᴬ[i, j, k]
@inline θ_and_sᴬ(i, j, k, θ::Number,       sᴬ::ArrayOrField) = @inbounds θ, sᴬ[i, j, k]
@inline θ_and_sᴬ(i, j, k, θ::ArrayOrField, sᴬ::Number)       = @inbounds θ[i, j, k], sᴬ
@inline θ_and_sᴬ(i, j, k, θ::Number,       sᴬ::Number)       = @inbounds θ, sᴬ

# Basic functionality
@inline ρ′(i, j, k, grid, eos, θ, sᴬ) = @inbounds ρ′(θ_and_sᴬ(i, j, k, θ, sᴬ)..., Zᵃᵃᶜ(i, j, k, grid), eos)

@inline thermal_expansionᶜᶜᶜ(i, j, k, grid, eos, θ, sᴬ) = @inbounds thermal_expansion(θ_and_sᴬ(i, j, k, θ, sᴬ)..., Zᵃᵃᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶠᶜᶜ(i, j, k, grid, eos, θ, sᴬ) = @inbounds thermal_expansion(ℑxᶠᵃᵃ(i, j, k, grid, θ), ℑxᶠᵃᵃ(i, j, k, grid, sᴬ), Zᵃᵃᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶜᶠᶜ(i, j, k, grid, eos, θ, sᴬ) = @inbounds thermal_expansion(ℑyᵃᶠᵃ(i, j, k, grid, θ), ℑyᵃᶠᵃ(i, j, k, grid, sᴬ), Zᵃᵃᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶜᶜᶠ(i, j, k, grid, eos, θ, sᴬ) = @inbounds thermal_expansion(ℑzᵃᵃᶠ(i, j, k, grid, θ), ℑzᵃᵃᶠ(i, j, k, grid, sᴬ), Zᵃᵃᶠ(i, j, k, grid), eos)

@inline haline_contractionᶜᶜᶜ(i, j, k, grid, eos, θ, sᴬ) = @inbounds haline_contraction(θ_and_sᴬ(i, j, k, θ, sᴬ)..., Zᵃᵃᶜ(i, j, k, grid), eos)
@inline haline_contractionᶠᶜᶜ(i, j, k, grid, eos, θ, sᴬ) = @inbounds haline_contraction(ℑxᶠᵃᵃ(i, j, k, grid, θ), ℑxᶠᵃᵃ(i, j, k, grid, sᴬ), Zᵃᵃᶜ(i, j, k, grid), eos)
@inline haline_contractionᶜᶠᶜ(i, j, k, grid, eos, θ, sᴬ) = @inbounds haline_contraction(ℑyᵃᶠᵃ(i, j, k, grid, θ), ℑyᵃᶠᵃ(i, j, k, grid, sᴬ), Zᵃᵃᶜ(i, j, k, grid), eos)
@inline haline_contractionᶜᶜᶠ(i, j, k, grid, eos, θ, sᴬ) = @inbounds haline_contraction(ℑzᵃᵃᶠ(i, j, k, grid, θ), ℑzᵃᵃᶠ(i, j, k, grid, sᴬ), Zᵃᵃᶠ(i, j, k, grid), eos)
