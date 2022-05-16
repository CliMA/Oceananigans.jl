
const ω̂₁ = 5/18 
const ω̂ₙ = 5/18  

const ε₂ = 1e-20

@inline function div_Uc(i, j, k, grid, advection::PositiveWENO, U, c)

    lower_limit = advection.bounds[1]
    upper_limit = advection.bounds[2]

    cᵢⱼ  = c[i, j, k]

    cᵢ₊ᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, c)
    cᵢ₊ᴿ = _right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, c)
    cᵢ₋ᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, c)
    cᵢ₋ᴿ = _right_biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, c)

    p̃ᵢ   =  (cᵢⱼ - ω̂₁ * cᵢ₋ᴿ - ω̂ₙ * cᵢ₊ᴸ) / (1 - 2ω̂₁)

    M    = max(p̃ᵢ, cᵢ₊ᴸ, cᵢ₋ᴿ) 
    m    = min(p̃ᵢ, cᵢ₊ᴸ, cᵢ₋ᴿ) 
    θ    = min(abs((upper_limit - cᵢⱼ)/(M - cᵢⱼ + ε₂)), abs((lower_limit - cᵢⱼ)/(m - cᵢⱼ + ε₂)), 1.0)
    
    cᵢ₊ᴸ = θ * (cᵢ₊ᴸ - cᵢⱼ) + cᵢⱼ
    cᵢ₋ᴿ = θ * (cᵢ₋ᴿ - cᵢⱼ) + cᵢⱼ

    flux_div_x = Axᶠᶜᶜ(i+1, j, k, grid) * upwind_biased_product(U.u[i+1, j, k], cᵢ₊ᴸ, cᵢ₊ᴿ) - 
                 Axᶠᶜᶜ(i,   j, k, grid) * upwind_biased_product(U.u[i,   j, k], cᵢ₋ᴸ, cᵢ₋ᴿ)

    cⱼ₊ᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, c)
    cⱼ₊ᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, c)
    cⱼ₋ᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, c)
    cⱼ₋ᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, c)

    p̃ⱼ   =  (cᵢⱼ - ω̂₁ * cⱼ₋ᴿ - ω̂ₙ * cⱼ₊ᴸ) / (1 - 2ω̂₁)

    M    = max(p̃ⱼ, cⱼ₊ᴸ, cⱼ₋ᴿ) 
    m    = min(p̃ⱼ, cⱼ₊ᴸ, cⱼ₋ᴿ) 
    θ    = min(abs((upper_limit - cᵢⱼ)/(M - cᵢⱼ + ε₂)), abs((lower_limit - cᵢⱼ)/(m - cᵢⱼ + ε₂)), 1.0)

    cⱼ₊ᴸ = θ * (cⱼ₊ᴸ - cᵢⱼ) + cᵢⱼ
    cⱼ₋ᴿ = θ * (cⱼ₋ᴿ - cᵢⱼ) + cᵢⱼ

    flux_div_y = Ayᶜᶠᶜ(i, j+1, k, grid) * upwind_biased_product(U.v[i, j+1, k], cⱼ₊ᴸ, cⱼ₊ᴿ) - 
                 Ayᶜᶠᶜ(i, j,   k, grid) * upwind_biased_product(U.v[i, j,   k], cⱼ₋ᴸ, cⱼ₋ᴿ)

    return 1/Vᶜᶜᶜ(i, j, k, grid) * (flux_div_x + flux_div_y + δzᵃᵃᶜ(i, j, k, grid, advective_tracer_flux_z, advection, U.w, c))
end
