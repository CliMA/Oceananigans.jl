
const ω̂₁ = 5/18 
const ω̂ₙ = 5/18  

const ε₂ = 1e-20

@inline function div_Uc(i, j, k, grid, advection::PositiveWENO, U, c)

    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, U.v, c)

    return 1/Vᶜᶜᶜ(i, j, k, grid) * (flux_div_x + flux_div_y + δzᵃᵃᶜ(i, j, k, grid, advective_tracer_flux_z, advection, U.w, c))
end

@inline function bounded_tracer_flux_divergence_x(i, j, k, grid, advection::PositiveWENO, u, c)

    lower_limit = advection.bounds[1]
    upper_limit = advection.bounds[2]

    cᵢⱼ  = c[i, j, k]

    c₊ᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, c)
    c₊ᴿ = _right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, advection, c)
    c₋ᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, c)
    c₋ᴿ = _right_biased_interpolate_xᶠᵃᵃ(i,   j, k, grid, advection, c)

    p̃   =  (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M   = max(p̃, c₊ᴸ, c₋ᴿ) 
    m   = min(p̃, c₊ᴸ, c₋ᴿ) 
    θ   = min(abs((upper_limit - cᵢⱼ)/(M - cᵢⱼ + ε₂)), abs((lower_limit - cᵢⱼ)/(m - cᵢⱼ + ε₂)), 1.0)
    
    c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    return Axᶠᶜᶜ(i+1, j, k, grid) * upwind_biased_product(u[i+1, j, k], c₊ᴸ, c₊ᴿ) - 
           Axᶠᶜᶜ(i,   j, k, grid) * upwind_biased_product(u[i,   j, k], c₋ᴸ, c₋ᴿ)
end


@inline function bounded_tracer_flux_divergence_y(i, j, k, grid, advection::PositiveWENO, v, c)

    lower_limit = advection.bounds[1]
    upper_limit = advection.bounds[2]

    cᵢⱼ  = c[i, j, k]

    c₊ᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, c)
    c₊ᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, advection, c)
    c₋ᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, c)
    c₋ᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j,   k, grid, advection, c)

    p̃   =  (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M   = max(p̃, c₊ᴸ, c₋ᴿ) 
    m   = min(p̃, c₊ᴸ, c₋ᴿ) 
    θ   = min(abs((upper_limit - cᵢⱼ)/(M - cᵢⱼ + ε₂)), abs((lower_limit - cᵢⱼ)/(m - cᵢⱼ + ε₂)), 1.0)

    c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    return Ayᶜᶠᶜ(i, j+1, k, grid) * upwind_biased_product(v[i, j+1, k], c₊ᴸ, c₊ᴿ) - 
           Ayᶜᶠᶜ(i, j,   k, grid) * upwind_biased_product(v[i, j,   k], c₋ᴸ, c₋ᴿ)
end