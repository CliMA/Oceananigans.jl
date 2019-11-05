module Operators

export
    Δx, Δy, ΔzF, ΔzC,
    Axᵃᵃᶜ, Axᵃᵃᶠ, Ayᵃᵃᶜ, Ayᵃᵃᶠ, Azᵃᵃᵃ,
    Vᵃᵃᶜ, Vᵃᵃᶠ,
    δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ,
    Ax_ψᵃᵃᶠ, Ax_ψᵃᵃᶜ, Ay_ψᵃᵃᶠ, Ay_ψᵃᵃᶜ, Az_ψᵃᵃᵃ,
    δᴶxᶜᵃᶜ, δᴶxᶜᵃᶠ, δᴶxᶠᵃᶜ, δᴶxᶠᵃᶠ, δᴶyᵃᶜᶜ, δᴶyᵃᶜᶠ, δᴶyᵃᶠᶜ, δᴶyᵃᶠᶠ, δᴶzᵃᵃᶜ, δᴶzᵃᵃᶠ,
    ∂xᶜᵃᵃ, ∂xᶠᵃᵃ, ∂yᵃᶜᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶜ, ∂zᵃᵃᶠ,
    ∂²xᶜᵃᵃ, ∂²xᶠᵃᵃ, ∂²yᵃᶜᵃ, ∂²yᵃᶠᵃ, ∂²zᵃᵃᶜ, ∂²zᵃᵃᶠ,
    ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ,
    ℑᴶxᶜᵃᵃ, ℑᴶxᶠᵃᵃ, ℑᴶyᵃᶜᵃ, ℑᴶyᵃᶠᵃ, ℑᴶzᵃᵃᶜ, ℑᴶzᵃᵃᶠ,
    ℑxyᶜᶜᵃ, ℑxyᶠᶜᵃ, ℑxyᶠᶠᵃ, ℑxyᶜᶠᵃ, ℑxzᶜᵃᶜ, ℑxzᶠᵃᶜ, ℑxzᶠᵃᶠ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶜ, ℑyzᵃᶠᶜ, ℑyzᵃᶠᶠ, ℑyzᵃᶜ,
    ℑxyzᶠᶠᶜ, ℑxyzᶜᶜᶠ,
    hdivᶜᶜᵃ, divᶜᶜᶜ, ∇²,
    div_uc, div_ũu, div_ũv, div_ũw

include("areas_and_volumes.jl")
include("difference_operators.jl")
include("derivative_operators.jl")
include("interpolation_operators.jl")
include("divergence_operators.jl")
include("laplacian_operators.jl")

include("tracer_advection_operators.jl")
include("momentum_advection_operators.jl")

end
