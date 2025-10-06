using Oceananigans.TurbulenceClosures: Σ₁₁, Σ₂₂, Σ₃₃, Σ₁₂, Σ₁₃, Σ₂₃
using Oceananigans.TurbulenceClosures: tr_Σ², Σ₁₂², Σ₁₃², Σ₂₃²
using Oceananigans.Operators: volume

#####
##### Double dot product of strain on cell edges (currently unused)
#####

"Return the double dot product of strain at `ccc`."
@inline ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w) =      tr_Σ²(i, j, k, grid, u, v, w) +
                                            2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃², u, v, w)

"Return the double dot product of strain at `ffc`."
@inline ΣᵢⱼΣᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w) =     ℑxyᶠᶠᵃ(i, j, k, grid, tr_Σ², u, v, w) +
                                            2 *   Σ₁₂²(i, j, k, grid, u, v, w) +
                                            2 * ℑyzᵃᶠᶜ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 * ℑxzᶠᵃᶜ(i, j, k, grid, Σ₂₃², u, v, w)

"Return the double dot product of strain at `fcf`."
@inline ΣᵢⱼΣᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w) =     ℑxzᶠᵃᶠ(i, j, k, grid, tr_Σ², u, v, w) +
                                            2 * ℑyzᵃᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 *   Σ₁₃²(i, j, k, grid, u, v, w) +
                                            2 * ℑxyᶠᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)

"Return the double dot product of strain at `cff`."
@inline ΣᵢⱼΣᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w) =     ℑyzᵃᶠᶠ(i, j, k, grid, tr_Σ², u, v, w) +
                                            2 * ℑxzᶜᵃᶠ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 * ℑxyᶜᶠᵃ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 *   Σ₂₃²(i, j, k, grid, u, v, w)

"Return the double dot product of strain at `ccf`."
@inline ΣᵢⱼΣᵢⱼᶜᶜᶠ(i, j, k, grid, u, v, w) =       ℑzᵃᵃᶠ(i, j, k, grid, tr_Σ², u, v, w) +
                                            2 * ℑxyzᶜᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w) +
                                            2 *   ℑxᶜᵃᵃ(i, j, k, grid, Σ₁₃², u, v, w) +
                                            2 *   ℑyᵃᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)


#####
##### Filtering
#####

# Filter is equivalent to:
# @inline filter(i, j, k, grid, f, args...) = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, f, args...)

@inline filter(i, j, k, grid, u::AbstractArray) = @inbounds (6 * u[i, j, k] +
                                                            u[i+1, j, k] + u[i-1, j, k] +
                                                            u[i, j+1, k] + u[i, j-1, k] +
                                                            u[i, j, k+1] + u[i, j, k-1]) / 12

@inline filter(i, j, k, grid, f, args...) = (6 * f(i, j, k, grid, args...) +
                                             f(i+1, j, k, grid, args...) + f(i-1, j, k, grid, args...) +
                                             f(i, j+1, k, grid, args...) + f(i, j-1, k, grid, args...) +
                                             f(i, j, k+1, grid, args...) + f(i, j, k-1, grid, args...)) / 12

#####
##### Velocity gradients
#####

# Diagonal
@inline ∂x_ū(i, j, k, grid, u) = ∂xᶜᶜᶜ(i, j, k, grid, filter, u)
@inline ∂y_v̄(i, j, k, grid, v) = ∂yᶜᶜᶜ(i, j, k, grid, filter, v)
@inline ∂z_w̄(i, j, k, grid, w) = ∂zᶜᶜᶜ(i, j, k, grid, filter, w)

# Off-diagonal
@inline ∂x_v̄(i, j, k, grid, v) = ∂xᶠᶠᶜ(i, j, k, grid, filter, v)
@inline ∂x_w̄(i, j, k, grid, w) = ∂xᶠᶜᶠ(i, j, k, grid, filter, w)

@inline ∂y_ū(i, j, k, grid, u) = ∂yᶠᶠᶜ(i, j, k, grid, filter, u)
@inline ∂y_w̄(i, j, k, grid, w) = ∂yᶜᶠᶠ(i, j, k, grid, filter, w)

@inline ∂z_ū(i, j, k, grid, u) = ∂zᶠᶜᶠ(i, j, k, grid, filter, u)
@inline ∂z_v̄(i, j, k, grid, v) = ∂zᶜᶠᶠ(i, j, k, grid, filter, v)

#####
##### Strain components
#####

const AG = AbstractGrid

# ccc strain components
@inline Σ̄₁₁(i, j, k, grid, u) = ∂x_ū(i, j, k, grid, u)
@inline Σ̄₂₂(i, j, k, grid, v) = ∂y_v̄(i, j, k, grid, v)
@inline Σ̄₃₃(i, j, k, grid, w) = ∂z_w̄(i, j, k, grid, w)

@inline tr_Σ̄(i, j, k, grid, u, v, w) = Σ̄₁₁(i, j, k, grid, u) + Σ̄₂₂(i, j, k, grid, v) + Σ̄₃₃(i, j, k, grid, w)
@inline tr_Σ̄²(ijk...) = Σ̄₁₁(ijk...)^2 + Σ̄₂₂(ijk...)^2 + Σ̄₃₃(ijk...)^2

# ffc
@inline Σ̄₁₂(i, j, k, grid::AG{FT}, u, v) where FT = FT(0.5) * (∂y_ū(i, j, k, grid, u) + ∂x_v̄(i, j, k, grid, v))
@inline Σ̄₁₂²(i, j, k, grid, u, v) = Σ̄₁₂(i, j, k, grid, u, v)^2

# fcf
@inline Σ̄₁₃(i, j, k, grid::AG{FT}, u, w) where FT = FT(0.5) * (∂z_ū(i, j, k, grid, u) + ∂x_w̄(i, j, k, grid, w))
@inline Σ̄₁₃²(i, j, k, grid, u, w) = Σ̄₁₃(i, j, k, grid, u, w)^2

# cff
@inline Σ̄₂₃(i, j, k, grid::AG{FT}, v, w) where FT = FT(0.5) * (∂z_v̄(i, j, k, grid, v) + ∂y_w̄(i, j, k, grid, w))
@inline Σ̄₂₃²(i, j, k, grid, v, w) = Σ̄₂₃(i, j, k, grid, v, w)^2

@inline Σ̄₁₁(i, j, k, grid, u, v, w) = Σ̄₁₁(i, j, k, grid, u)
@inline Σ̄₂₂(i, j, k, grid, u, v, w) = Σ̄₂₂(i, j, k, grid, v)
@inline Σ̄₃₃(i, j, k, grid, u, v, w) = Σ̄₃₃(i, j, k, grid, w)

@inline Σ̄₁₂(i, j, k, grid, u, v, w) = Σ̄₁₂(i, j, k, grid, u, v)
@inline Σ̄₁₃(i, j, k, grid, u, v, w) = Σ̄₁₃(i, j, k, grid, u, w)
@inline Σ̄₂₃(i, j, k, grid, u, v, w) = Σ̄₂₃(i, j, k, grid, v, w)

@inline Σ̄₁₂²(i, j, k, grid, u, v, w) = Σ̄₁₂²(i, j, k, grid, u, v)
@inline Σ̄₁₃²(i, j, k, grid, u, v, w) = Σ̄₁₃²(i, j, k, grid, u, w)
@inline Σ̄₂₃²(i, j, k, grid, u, v, w) = Σ̄₂₃²(i, j, k, grid, v, w)

#####
##### Double dot product of strain on cell edges
#####

"Return the double dot product of strain at `ccc` on a 2δ test grid."
@inline Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w) = (tr_Σ̄²(i, j, k, grid, u, v, w)
                                             + 2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ̄₁₂², u, v, w)
                                             + 2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ̄₁₃², u, v, w)
                                             + 2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ̄₂₃², u, v, w))

# Here the notation ⟨A⟩ is equivalent to Ā: a filter of size 2Δᶠ, where Δᶠ is the grid scale.

@inline ΣΣ₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * Σ₁₁(i, j, k, grid, u, v, w)
@inline ΣΣ₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * Σ₂₂(i, j, k, grid, u, v, w)
@inline ΣΣ₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * Σ₃₃(i, j, k, grid, u, v, w)

@inline ΣΣ₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂, u, v, w)
@inline ΣΣ₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃, u, v, w)
@inline ΣΣ₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃, u, v, w)

@inline var"⟨ΣΣ₁₁⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = filter(i, j, k, grid, ΣΣ₁₁ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)
@inline var"⟨ΣΣ₂₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = filter(i, j, k, grid, ΣΣ₂₂ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)
@inline var"⟨ΣΣ₃₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = filter(i, j, k, grid, ΣΣ₃₃ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)

@inline var"⟨ΣΣ₁₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = filter(i, j, k, grid, ΣΣ₁₂ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)
@inline var"⟨ΣΣ₁₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = filter(i, j, k, grid, ΣΣ₁₃ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)
@inline var"⟨ΣΣ₂₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = filter(i, j, k, grid, ΣΣ₂₃ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)

@inline Σ̄Σ̄₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * Σ̄₁₁(i, j, k, grid, u, v, w)
@inline Σ̄Σ̄₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * Σ̄₂₂(i, j, k, grid, u, v, w)
@inline Σ̄Σ̄₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * Σ̄₃₃(i, j, k, grid, u, v, w)

@inline Σ̄Σ̄₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * ℑxyᶜᶜᵃ(i, j, k, grid, Σ̄₁₂, u, v, w)
@inline Σ̄Σ̄₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * ℑxzᶜᵃᶜ(i, j, k, grid, Σ̄₁₃, u, v, w)
@inline Σ̄Σ̄₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * ℑyzᵃᶜᶜ(i, j, k, grid, Σ̄₂₃, u, v, w)

const ᾱ² = 4
const β  = 1
@inline Δᶠ(i, j, k, grid) = ∛volume(i, j, k, grid, Center(), Center(), Center())
@inline M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨ΣΣ₁₁⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - ᾱ² * β * Σ̄Σ̄₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))
@inline M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨ΣΣ₂₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - ᾱ² * β * Σ̄Σ̄₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))
@inline M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨ΣΣ₃₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - ᾱ² * β * Σ̄Σ̄₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))

@inline M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨ΣΣ₁₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - ᾱ² * β * Σ̄Σ̄₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))
@inline M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨ΣΣ₁₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - ᾱ² * β * Σ̄Σ̄₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))
@inline M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨ΣΣ₂₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - ᾱ² * β * Σ̄Σ̄₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))

@inline uᵢ²(i, j, k, grid, uᵢ) = @inbounds uᵢ[i, j, k]^2
@inline u₁u₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, uᵢ², u)
@inline u₂u₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, uᵢ², v)
@inline u₃u₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑzᵃᵃᶜ(i, j, k, grid, uᵢ², w)

@inline ū²(i, j, k, grid, u) = filter(i, j, k, grid, u)^2
@inline v̄²(i, j, k, grid, v) = filter(i, j, k, grid, v)^2
@inline w̄²(i, j, k, grid, w) = filter(i, j, k, grid, w)^2

@inline ū₁ū₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ū², u)
@inline ū₂ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, v̄², v)
@inline ū₃ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑzᵃᵃᶜ(i, j, k, grid, w̄², w)

@inline u₁u₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, u) * ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline u₁u₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, u) * ℑzᵃᵃᶜ(i, j, k, grid, w)
@inline u₂u₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, v) * ℑzᵃᵃᶜ(i, j, k, grid, w)

@inline ū₁ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, filter, u) * ℑyᵃᶜᵃ(i, j, k, grid, filter, v)
@inline ū₁ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, filter, u) * ℑzᵃᵃᶜ(i, j, k, grid, filter, w)
@inline ū₂ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, filter, v) * ℑzᵃᵃᶜ(i, j, k, grid, filter, w)

@inline L₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = filter(i, j, k, grid, u₁u₁ᶜᶜᶜ, u, v, w) - ū₁ū₁ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = filter(i, j, k, grid, u₂u₂ᶜᶜᶜ, u, v, w) - ū₂ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = filter(i, j, k, grid, u₃u₃ᶜᶜᶜ, u, v, w) - ū₃ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)

@inline L₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = filter(i, j, k, grid, u₁u₂ᶜᶜᶜ, u, v, w) - ū₁ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = filter(i, j, k, grid, u₁u₃ᶜᶜᶜ, u, v, w) - ū₁ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = filter(i, j, k, grid, u₂u₃ᶜᶜᶜ, u, v, w) - ū₂ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)

