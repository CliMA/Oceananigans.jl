
const ω̂₁ = 5/18 
const ω̂ₙ = 5/18  

const ε₂ = 1e-20

# Here in the future we can easily add UpwindBiasedFifthOrder 
const BoundPreservingScheme = PositiveWENO

@inline function div_Uc(i, j, k, grid, advection::BoundPreservingScheme, U, c, val_tracer_index)

    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, U.u, c, val_tracer_index)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, U.v, c, val_tracer_index)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, U.w, c, val_tracer_index)

    return 1/Vᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

@inline function bounded_tracer_flux_divergence_x(i, j, k, grid, advection::BoundPreservingScheme, u, c, val_tracer_index)

    lower_limit = advection.bounds[val_tracer_index][1]
    upper_limit = advection.bounds[val_tracer_index][2]

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

@inline function bounded_tracer_flux_divergence_y(i, j, k, grid, advection::BoundPreservingScheme, v, c, val_tracer_index)

    lower_limit = advection.bounds[val_tracer_index][1]
    upper_limit = advection.bounds[val_tracer_index][2]

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

@inline function bounded_tracer_flux_divergence_z(i, j, k, grid, advection::BoundPreservingScheme, w, c, val_tracer_index)

    lower_limit = advection.bounds[val_tracer_index][1]
    upper_limit = advection.bounds[val_tracer_index][2]

    cᵢⱼ  = c[i, j, k]

    c₊ᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, c)
    c₊ᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, advection, c)
    c₋ᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, c)
    c₋ᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k,   grid, advection, c)

    p̃   =  (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M   = max(p̃, c₊ᴸ, c₋ᴿ) 
    m   = min(p̃, c₊ᴸ, c₋ᴿ) 
    θ   = min(abs((upper_limit - cᵢⱼ)/(M - cᵢⱼ + ε₂)), abs((lower_limit - cᵢⱼ)/(m - cᵢⱼ + ε₂)), 1.0)

    c₊ᴸ = θ * (c₊ᴸ - cᵢⱼ) + cᵢⱼ
    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    return Azᶜᶜᶠ(i, j, k+1, grid) * upwind_biased_product(w[i, j, k+1], c₊ᴸ, c₊ᴿ) - 
           Azᶜᶜᶠ(i, j, k,   grid) * upwind_biased_product(w[i, j, k],   c₋ᴸ, c₋ᴿ)
end