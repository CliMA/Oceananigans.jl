@inline ∇_dot_qᶜ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)
@inline ∂ⱼ_τ₁ⱼ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)
@inline ∂ⱼ_τ₂ⱼ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)
@inline ∂ⱼ_τ₃ⱼ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)

calculate_diffusivities!(diffusivities, ::Nothing, args...; kwargs...) = nothing
calculate_diffusivities!(::Nothing, ::Nothing, args...; kwargs...) = nothing

@inline viscosity(::Nothing, ::Nothing) = 0
@inline diffusivity(::Nothing, ::Nothing, ::Val{id}) where id = 0