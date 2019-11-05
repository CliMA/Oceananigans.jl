"""
    ν_σᶜᶜᶜ(i, j, k, grid, ν, σᶜᶜᶜ, u, v, w)

Multiply the array `ν` located at `ᶜᶜᶜ` by a function

    `σᶜᶜᶜ(i, j, k, grid, u, v, w)`

at index `i, j, k` and location `ᶜᶜᶜ`.
"""
@inline ν_σᶜᶜᶜ(i, j, k, grid, ν::TN, σᶜᶜᶜ::TS, u, v, w) where {TN<:AbstractArray, TS} =
    @inbounds ν[i, j, k] * σᶜᶜᶜ(i, j, k, grid, u, v, w)

"""
    ν_σᶠᶠᶜ(i, j, k, grid, ν, σᶠᶠᶜ, u, v, w)

Multiply the array `ν` located at `ᶜᶜᶜ` by a function

    `σᶠᶠᶜ(i, j, k, grid, u, v, w)`

at index `i, j, k` and location `ᶠᶠᶜ`.
"""
@inline ν_σᶠᶠᶜ(i, j, k, grid, ν::TN, σᶠᶠᶜ::TS, u, v, w) where {TN<:AbstractArray, TS} =
    @inbounds ℑxyᶠᶠᵃ(i, j, k, grid, ν) * σᶠᶠᶜ(i, j, k, grid, u, v, w)

# These functions are analogous to the two above, but for different locations:
@inline ν_σᶠᶜᶠ(i, j, k, grid, ν::TN, σᶠᶜᶠ::TS, u, v, w) where {TN<:AbstractArray, TS} =
    @inbounds ℑxzᶠᵃᶠ(i, j, k, grid, ν) * σᶠᶜᶠ(i, j, k, grid, u, v, w)

@inline ν_σᶜᶠᶠ(i, j, k, grid, ν::TN, σᶜᶠᶠ::TS, u, v, w) where {TN<:AbstractArray, TS} =
    @inbounds ℑyzᵃᶠᶠ(i, j, k, grid, ν) * σᶜᶠᶠ(i, j, k, grid, u, v, w)

#####
##### Stress divergences
#####

# At fcc
@inline ∂x_2ν_Σ₁₁(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂xᶠᵃᵃ(i, j, k, grid, ν_σᶜᶜᶜ, diffusivities.νₑ, Σ₁₁, U.u, U.v, U.w)

@inline ∂y_2ν_Σ₁₂(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂yᵃᶜᵃ(i, j, k, grid, ν_σᶠᶠᶜ, diffusivities.νₑ, Σ₁₂, U.u, U.v, U.w)

@inline ∂z_2ν_Σ₁₃(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂zᵃᵃᶜ(i, j, k, grid, ν_σᶠᶜᶠ, diffusivities.νₑ, Σ₁₃, U.u, U.v, U.w)

# At cfc
@inline ∂x_2ν_Σ₂₁(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂xᶜᵃᵃ(i, j, k, grid, ν_σᶠᶠᶜ, diffusivities.νₑ, Σ₂₁, U.u, U.v, U.w)

@inline ∂y_2ν_Σ₂₂(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂yᵃᶠᵃ(i, j, k, grid, ν_σᶜᶜᶜ, diffusivities.νₑ, Σ₂₂, U.u, U.v, U.w)

@inline ∂z_2ν_Σ₂₃(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂zᵃᵃᶜ(i, j, k, grid, ν_σᶜᶠᶠ, diffusivities.νₑ, Σ₂₃, U.u, U.v, U.w)

# At ccf
@inline ∂x_2ν_Σ₃₁(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂xᶜᵃᵃ(i, j, k, grid, ν_σᶠᶜᶠ, diffusivities.νₑ, Σ₃₁, U.u, U.v, U.w)

@inline ∂y_2ν_Σ₃₂(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂yᵃᶜᵃ(i, j, k, grid, ν_σᶜᶠᶠ, diffusivities.νₑ, Σ₃₂, U.u, U.v, U.w)

@inline ∂z_2ν_Σ₃₃(i, j, k, grid, closure, U, diffusivities) =
    2 * ∂zᵃᵃᶠ(i, j, k, grid, ν_σᶜᶜᶜ, diffusivities.νₑ, Σ₃₃, U.u, U.v, U.w)

"""
    κ_∂x_c(i, j, k, grid, c, κ, closure, args...)

Return `κ ∂x c`, where `κ` is an array or function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂x_c(i, j, k, grid, κ, c, closure, args...)
    κ = ℑxᶠᵃᵃ(i, j, k, grid, κ, closure, args...)
    ∂x_c = ∂xᶠᵃᵃ(i, j, k, grid, c)
    return κ * ∂x_c
end

"""
    κ_∂y_c(i, j, k, grid, c, κ, closure, args...)

Return `κ ∂y c`, where `κ` is an array or function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂y_c(i, j, k, grid, κ, c, closure, args...)
    κ = ℑyᵃᶠᵃ(i, j, k, grid, κ, closure, args...)
    ∂y_c = ∂yᵃᶠᵃ(i, j, k, grid, c)
    return κ * ∂y_c
end

"""
    κ_∂z_c(i, j, k, grid, c, κ, closure, buoyancy, u, v, w, T, S)

Return `κ ∂z c`, where `κ` is an array or function that computes
diffusivity at cell centers (location `ccc`), and `c` is an array of scalar
data located at cell centers.
"""
@inline function κ_∂z_c(i, j, k, grid, κ, c, closure, args...)
    κ = ℑzᵃᵃᶠ(i, j, k, grid, κ, closure, args...)
    ∂z_c = ∂zᵃᵃᶠ(i, j, k, grid, c)
    return κ * ∂z_c
end

"""
    ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, U, diffusivities)

Return the ``x``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₁₁) + ∂y(2 ν Σ₁₁) + ∂z(2 ν Σ₁₁)`

at the location `fcc`.
"""
@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::IsotropicViscosity, U, diffusivities) = (
      ∂x_2ν_Σ₁₁(i, j, k, grid, closure, U, diffusivities)
    + ∂y_2ν_Σ₁₂(i, j, k, grid, closure, U, diffusivities)
    + ∂z_2ν_Σ₁₃(i, j, k, grid, closure, U, diffusivities)
)

"""
    ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure, U, diffusivities)

Return the ``y``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₂₁) + ∂y(2 ν Σ₂₂) + ∂z(2 ν Σ₂₂)`

at the location `ccf`.
"""
@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::IsotropicViscosity, U, diffusivities) = (
      ∂x_2ν_Σ₂₁(i, j, k, grid, closure, U, diffusivities)
    + ∂y_2ν_Σ₂₂(i, j, k, grid, closure, U, diffusivities)
    + ∂z_2ν_Σ₂₃(i, j, k, grid, closure, U, diffusivities)
)

"""
    ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure, diffusivities)

Return the ``z``-component of the turbulent diffusive flux divergence:

`∂x(2 ν Σ₃₁) + ∂y(2 ν Σ₃₂) + ∂z(2 ν Σ₃₃)`

at the location `ccf`.
"""
@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::IsotropicViscosity, U, diffusivities) = (
      ∂x_2ν_Σ₃₁(i, j, k, grid, closure, U, diffusivities)
    + ∂y_2ν_Σ₃₂(i, j, k, grid, closure, U, diffusivities)
    + ∂z_2ν_Σ₃₃(i, j, k, grid, closure, U, diffusivities)
)
