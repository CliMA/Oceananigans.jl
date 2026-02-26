@inline ∇_dot_qᶜ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)
@inline ∂ⱼ_τ₁ⱼ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)
@inline ∂ⱼ_τ₂ⱼ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)
@inline ∂ⱼ_τ₃ⱼ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)

compute_closure_fields!(closure_fields, ::Nothing, args...; kwargs...) = nothing
compute_closure_fields!(::Nothing, ::Nothing, args...; kwargs...) = nothing

step_closure_prognostics!(closure_fields, ::Nothing, args...) = nothing
step_closure_prognostics!(::Nothing, ::Nothing, args...) = nothing

@inline viscosity(::Nothing, ::Nothing) = 0
@inline diffusivity(::Nothing, ::Nothing, ::Val{id}) where id = 0
