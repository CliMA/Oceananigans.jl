""" Return the geopotential depth at `i, j, k` at cell centers. """
@inline Dᵃᵃᶜ(i, j, k, grid::AbstractGrid{FT}) where FT = @inbounds k < 1       ? - grid.zC[1] + (1 - k) * grid.Δz :
                                                                   k > grid.Nz ? - grid.zC[end] - (k - grid.Nz) * grid.Δz : 
                                                                                 - grid.zC[k]

""" Return the geopotential depth at `i, j, k` at cell z-interfaces. """
@inline Dᵃᵃᶠ(i, j, k, grid::AbstractGrid{FT}) where FT = @inbounds k < 1           ? - grid.zF[1] + (1 - k) * grid.Δz :
                                                                   k > grid.Nz + 1 ? - grid.zF[end] - (k - grid.Nz + 1) * grid.Δz : 
                                                                                     - grid.zF[k]
# Dispatch shenanigans
@inline θ_and_sᴬ(i, j, k, θ::AbstractArray, sᴬ::AbstractArray) = @inbounds θ[i, j, k], sᴬ[i, j, k]
@inline θ_and_sᴬ(i, j, k, θ::Number,        sᴬ::AbstractArray) = @inbounds θ, sᴬ[i, j, k]
@inline θ_and_sᴬ(i, j, k, θ::AbstractArray, sᴬ::Number)        = @inbounds θ[i, j, k], sᴬ
@inline θ_and_sᴬ(i, j, k, θ::Number,        sᴬ::Number)        = @inbounds θ, sᴬ

# Basic functionality
@inline ρ′(i, j, k, grid, eos, θ, sᴬ) = @inbounds ρ′(θ_and_sᴬ(i, j, k, θ, sᴬ)..., Dᵃᵃᶜ(i, j, k, grid), eos)

@inline thermal_expansionᶜᶜᶜ(i, j, k, grid, eos, θ, sᴬ) = @inbounds thermal_expansion(θ_and_sᴬ(i, j, k, θ, sᴬ)..., Dᵃᵃᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶠᶜᶜ(i, j, k, grid, eos, θ, sᴬ) = @inbounds thermal_expansion(ℑxᶠᵃᵃ(i, j, k, grid, θ), ℑxᶠᵃᵃ(i, j, k, grid, sᴬ), Dᵃᵃᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶜᶠᶜ(i, j, k, grid, eos, θ, sᴬ) = @inbounds thermal_expansion(ℑyᵃᶠᵃ(i, j, k, grid, θ), ℑyᵃᶠᵃ(i, j, k, grid, sᴬ), Dᵃᵃᶜ(i, j, k, grid), eos)
@inline thermal_expansionᶜᶜᶠ(i, j, k, grid, eos, θ, sᴬ) = @inbounds thermal_expansion(ℑzᵃᵃᶠ(i, j, k, grid, θ), ℑzᵃᵃᶠ(i, j, k, grid, sᴬ), Dᵃᵃᶠ(i, j, k, grid), eos)

@inline haline_contractionᶜᶜᶜ(i, j, k, grid, eos, θ, sᴬ) = @inbounds haline_contraction(θ_and_sᴬ(i, j, k, θ, sᴬ)..., Dᵃᵃᶜ(i, j, k, grid), eos)
@inline haline_contractionᶠᶜᶜ(i, j, k, grid, eos, θ, sᴬ) = @inbounds haline_contraction(ℑxᶠᵃᵃ(i, j, k, grid, θ), ℑxᶠᵃᵃ(i, j, k, grid, sᴬ), Dᵃᵃᶜ(i, j, k, grid), eos)
@inline haline_contractionᶜᶠᶜ(i, j, k, grid, eos, θ, sᴬ) = @inbounds haline_contraction(ℑyᵃᶠᵃ(i, j, k, grid, θ), ℑyᵃᶠᵃ(i, j, k, grid, sᴬ), Dᵃᵃᶜ(i, j, k, grid), eos)
@inline haline_contractionᶜᶜᶠ(i, j, k, grid, eos, θ, sᴬ) = @inbounds haline_contraction(ℑzᵃᵃᶠ(i, j, k, grid, θ), ℑzᵃᵃᶠ(i, j, k, grid, sᴬ), Dᵃᵃᶠ(i, j, k, grid), eos)
