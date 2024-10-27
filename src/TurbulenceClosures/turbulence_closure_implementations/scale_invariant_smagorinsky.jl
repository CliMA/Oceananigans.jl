#####
##### In this version of the Smagorinsky closure, the coefficient is dynamically calculated but it's assumed to be invariant
##### with scale. Hence the name Scale-Invariant Smagorinsky. This a type of "dynamic Smagorinsky" closures.
#####

using Oceananigans.Operators: volume
using Statistics: mean!

#####
##### Filters
#####

# TODO: Generalize filter to stretched directions
const AG{FT} = AbstractGrid{FT} where FT

@inline ℱx²ᵟ(i, j, k, grid::AG{FT}, ϕ) where FT = @inbounds FT(0.5) * ϕ[i, j, k] + FT(0.25) * (ϕ[i-1, j, k] + ϕ[i+1, j,  k])
@inline ℱy²ᵟ(i, j, k, grid::AG{FT}, ϕ) where FT = @inbounds FT(0.5) * ϕ[i, j, k] + FT(0.25) * (ϕ[i, j-1, k] + ϕ[i,  j+1, k])
@inline ℱz²ᵟ(i, j, k, grid::AG{FT}, ϕ) where FT = @inbounds FT(0.5) * ϕ[i, j, k] + FT(0.25) * (ϕ[i, j, k-1] + ϕ[i,  j, k+1])

@inline ℱx²ᵟ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} =
    FT(0.5) * f(i, j, k, grid, args...) + FT(0.25) * (f(i-1, j, k, grid, args...) + f(i+1, j, k, grid, args...))

@inline ℱy²ᵟ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} =
    FT(0.5) * f(i, j, k, grid, args...) + FT(0.25) * (f(i, j-1, k, grid, args...) + f(i, j+1, k, grid, args...))

@inline ℱz²ᵟ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} =
    FT(0.5) * f(i, j, k, grid, args...) + FT(0.25) * (f(i, j, k-1, grid, args...) + f(i, j, k+1, grid, args...))

@inline ℱxy²ᵟ(i, j, k, grid, f, args...)  = ℱy²ᵟ(i, j, k, grid, ℱx²ᵟ, f, args...)
@inline ℱyz²ᵟ(i, j, k, grid, f, args...)  = ℱz²ᵟ(i, j, k, grid, ℱy²ᵟ, f, args...)
@inline ℱxz²ᵟ(i, j, k, grid, f, args...)  = ℱz²ᵟ(i, j, k, grid, ℱz²ᵟ, f, args...)
@inline ℱ²ᵟ(i, j, k, grid, f, args...)    = ℱz²ᵟ(i, j, k, grid, ℱxy²ᵟ, f, args...)

#####
##### Velocity gradients
#####

# Diagonal
@inline ∂x_ū(i, j, k, grid, u) = ∂xᶜᶜᶜ(i, j, k, grid, ℱ²ᵟ, u)
@inline ∂y_v̄(i, j, k, grid, v) = ∂yᶜᶜᶜ(i, j, k, grid, ℱ²ᵟ, v)
@inline ∂z_w̄(i, j, k, grid, w) = ∂zᶜᶜᶜ(i, j, k, grid, ℱ²ᵟ, w)

# Off-diagonal
@inline ∂x_v̄(i, j, k, grid, v) = ∂xᶠᶠᶜ(i, j, k, grid, ℱ²ᵟ, v)
@inline ∂x_w̄(i, j, k, grid, w) = ∂xᶠᶜᶠ(i, j, k, grid, ℱ²ᵟ, w)

@inline ∂y_ū(i, j, k, grid, u) = ∂yᶠᶠᶜ(i, j, k, grid, ℱ²ᵟ, u)
@inline ∂y_w̄(i, j, k, grid, w) = ∂yᶜᶠᶠ(i, j, k, grid, ℱ²ᵟ, w)

@inline ∂z_ū(i, j, k, grid, u) = ∂zᶠᶜᶠ(i, j, k, grid, ℱ²ᵟ, u)
@inline ∂z_v̄(i, j, k, grid, v) = ∂zᶜᶠᶠ(i, j, k, grid, ℱ²ᵟ, v)

#####
##### Strain components
#####

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

@inline SS₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * Σ₁₁(i, j, k, grid, u, v, w)
@inline SS₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * Σ₂₂(i, j, k, grid, u, v, w)
@inline SS₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * Σ₃₃(i, j, k, grid, u, v, w)

@inline SS₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂, u, v, w)
@inline SS₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃, u, v, w)
@inline SS₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = @inbounds Σᶜᶜᶜ[i, j, k] * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃, u, v, w)

@inline var"⟨SS₁₁⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = ℱ²ᵟ(i, j, k, grid, SS₁₁ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)
@inline var"⟨SS₂₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = ℱ²ᵟ(i, j, k, grid, SS₂₂ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)
@inline var"⟨SS₃₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = ℱ²ᵟ(i, j, k, grid, SS₃₃ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)

@inline var"⟨SS₁₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = ℱ²ᵟ(i, j, k, grid, SS₁₂ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)
@inline var"⟨SS₁₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = ℱ²ᵟ(i, j, k, grid, SS₁₃ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)
@inline var"⟨SS₂₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) = ℱ²ᵟ(i, j, k, grid, SS₂₃ᶜᶜᶜ, u, v, w, Σᶜᶜᶜ)

@inline S̄S̄₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * Σ̄₁₁(i, j, k, grid, u, v, w)
@inline S̄S̄₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * Σ̄₂₂(i, j, k, grid, u, v, w)
@inline S̄S̄₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * Σ̄₃₃(i, j, k, grid, u, v, w)

@inline S̄S̄₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * ℑxyᶜᶜᵃ(i, j, k, grid, Σ̄₁₂, u, v, w)
@inline S̄S̄₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * ℑxzᶜᵃᶜ(i, j, k, grid, Σ̄₁₃, u, v, w)
@inline S̄S̄₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ) = @inbounds Σ̄ᶜᶜᶜ[i, j, k] * ℑyzᵃᶜᶜ(i, j, k, grid, Σ̄₂₃, u, v, w)

@inline Δᶠ(i, j, k, grid) = ∛volume(i, j, k, grid, Center(), Center(), Center())
@inline M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨SS₁₁⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - α^2*β * S̄S̄₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))
@inline M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨SS₂₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - α^2*β * S̄S̄₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))
@inline M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨SS₃₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - α^2*β * S̄S̄₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))

@inline M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨SS₁₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - α^2*β * S̄S̄₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))
@inline M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨SS₁₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - α^2*β * S̄S̄₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))
@inline M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β, Σᶜᶜᶜ, Σ̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨SS₂₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Σᶜᶜᶜ) - α^2*β * S̄S̄₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Σ̄ᶜᶜᶜ))

@inline ϕψ(i, j, k, grid, ϕ, ψ) = @inbounds ϕ[i, j, k] * ψ[i, j, k]
@inline u₁u₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ϕψ, u, u)
@inline u₂u₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, ϕψ, v, v)
@inline u₃u₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑzᵃᵃᶜ(i, j, k, grid, ϕψ, w, w)

@inline ϕ̄ψ̄(i, j, k, grid, ϕ, ψ) = ℱ²ᵟ(i, j, k, grid, ϕ) * ℱ²ᵟ(i, j, k, grid, ψ)
@inline ū₁ū₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ϕ̄ψ̄, u, u)
@inline ū₂ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ϕ̄ψ̄, v, v)
@inline ū₃ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ϕ̄ψ̄, w, w)

@inline u₁u₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, u) * ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline u₁u₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, u) * ℑzᵃᵃᶜ(i, j, k, grid, w)
@inline u₂u₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, v) * ℑzᵃᵃᶜ(i, j, k, grid, w)

@inline ū₁ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ℱ²ᵟ, u) * ℑyᵃᶜᵃ(i, j, k, grid, ℱ²ᵟ, v)
@inline ū₁ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ℱ²ᵟ, u) * ℑzᵃᵃᶜ(i, j, k, grid, ℱ²ᵟ, w)
@inline ū₂ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, ℱ²ᵟ, v) * ℑzᵃᵃᶜ(i, j, k, grid, ℱ²ᵟ, w)

@inline L₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₁u₁ᶜᶜᶜ, u, v, w) - ū₁ū₁ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₂u₂ᶜᶜᶜ, u, v, w) - ū₂ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₃u₃ᶜᶜᶜ, u, v, w) - ū₃ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)

@inline L₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₁u₂ᶜᶜᶜ, u, v, w) - ū₁ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₁u₃ᶜᶜᶜ, u, v, w) - ū₁ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₂u₃ᶜᶜᶜ, u, v, w) - ū₂ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)
