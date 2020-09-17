@inline ∇_κ_∇c(i, j, k, grid::AbstractGrid{FT}, clock, closure::Nothing, args...) where FT = zero(FT)
@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid::AbstractGrid{FT}, clock, closure::Nothing, args...) where FT = zero(FT)
@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid::AbstractGrid{FT}, clock, closure::Nothing, args...) where FT = zero(FT)
@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid::AbstractGrid{FT}, clock, closure::Nothing, args...) where FT = zero(FT)

calculate_diffusivities!(K, arch, grid, closure::Nothing, args...) = nothing
