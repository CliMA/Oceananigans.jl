#####
##### Weighted Essentially Non-Oscillatory (WENO) scheme of order 5
#####

struct WENO5 <: AbstractAdvectionScheme end

#####
##### ENO interpolants of size 3
#####

@inline px₀(i, j, k, ψ) = @inbounds + 1/3 * ψ[i-1, j, k] + 5/6 * ψ[i,   j, k] -  1/6 * ψ[i+1, j, k]
@inline px₁(i, j, k, ψ) = @inbounds - 1/6 * ψ[i-2, j, k] + 5/6 * ψ[i-1, j, k] +  1/3 * ψ[i,   j, k]
@inline px₂(i, j, k, ψ) = @inbounds + 1/3 * ψ[i-3, j, k] - 7/6 * ψ[i-2, j, k] + 11/6 * ψ[i-1, j, k]

@inline py₀(i, j, k, ψ) = @inbounds + 1/3 * ψ[i, j-1, k] + 5/6 * ψ[i, j,   k] -  1/6 * ψ[i, j+1, k]
@inline py₁(i, j, k, ψ) = @inbounds - 1/6 * ψ[i, j-2, k] + 5/6 * ψ[i, j-1, k] +  1/3 * ψ[i, j  , k]
@inline py₂(i, j, k, ψ) = @inbounds + 1/3 * ψ[i, j-3, k] - 7/6 * ψ[i, j-2, k] + 11/6 * ψ[i, j-1, k]

@inline pz₀(i, j, k, ψ) = @inbounds + 1/3 * ψ[i, j, k-1] + 5/6 * ψ[i, j,   k] -  1/6 * ψ[i, j, k+1]
@inline pz₁(i, j, k, ψ) = @inbounds - 1/6 * ψ[i, j, k-2] + 5/6 * ψ[i, j, k-1] +  1/3 * ψ[i, j,   k]
@inline pz₂(i, j, k, ψ) = @inbounds + 1/3 * ψ[i, j, k-3] - 7/6 * ψ[i, j, k-2] + 11/6 * ψ[i, j, k-1]

#####
##### Jiang & Shu (1996) WENO smoothness indicators
#####

@inline βx₀(i, j, k, ψ) = @inbounds 13/12 * (ψ[i-1, j, k] - 2ψ[i,   j, k] + ψ[i+1, j, k])^2 + 1/4 * (3ψ[i-1, j, k] - 4ψ[i,   j, k] +  ψ[i+1, j, k])^2
@inline βx₁(i, j, k, ψ) = @inbounds 13/12 * (ψ[i-2, j, k] - 2ψ[i-1, j, k] + ψ[i,   j, k])^2 + 1/4 * ( ψ[i-2, j, k]                 -  ψ[i,   j, k])^2
@inline βx₂(i, j, k, ψ) = @inbounds 13/12 * (ψ[i-3, j, k] - 2ψ[i-2, j, k] + ψ[i-1, j, k])^2 + 1/4 * ( ψ[i-3, j, k] - 4ψ[i-2, j, k] + 3ψ[i-1, j, k])^2

@inline βy₀(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j-1, k] - 2ψ[i, j,   k] + ψ[i, j+1, k])^2 + 1/4 * (3ψ[i, j-1, k] - 4ψ[i,   j, k] +  ψ[i, j+1, k])^2
@inline βy₁(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j-2, k] - 2ψ[i, j-1, k] + ψ[i, j,   k])^2 + 1/4 * ( ψ[i, j-2, k]                 -  ψ[i,   j, k])^2
@inline βy₂(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j-3, k] - 2ψ[i, j-2, k] + ψ[i, j-1, k])^2 + 1/4 * ( ψ[i, j-3, k] - 4ψ[i, j-2, k] + 3ψ[i, j-1, k])^2

@inline βz₀(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j, k-1] - 2ψ[i, j,   k] + ψ[i, j, k+1])^2 + 1/4 * (3ψ[i, j, k-1] - 4ψ[i, j,   k] +  ψ[i, j, k+1])^2
@inline βz₁(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j, k-2] - 2ψ[i, j, k-1] + ψ[i, j,   k])^2 + 1/4 * ( ψ[i, j, k-2]                 -  ψ[i, j,   k])^2
@inline βz₂(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j, k-3] - 2ψ[i, j, k-2] + ψ[i, j, k-1])^2 + 1/4 * ( ψ[i, j, k-3] - 4ψ[i, j, k-2] + 3ψ[i, j, k-1])^2

#####
##### WENO-5 optimal weights
#####

const C3₀ = 3/10
const C3₁ = 3/5
const C3₂ = 1/10

#####
##### WENO-5 raw weights
#####

const ε = 1e-6
const ƞ = 2  # WENO exponent

@inline αx₀(i, j, k, ψ) = C3₀ / (βx₀(i, j, k, ψ) + ε)^ƞ
@inline αx₁(i, j, k, ψ) = C3₁ / (βx₁(i, j, k, ψ) + ε)^ƞ
@inline αx₂(i, j, k, ψ) = C3₂ / (βx₂(i, j, k, ψ) + ε)^ƞ

@inline αy₀(i, j, k, ψ) = C3₀ / (βy₀(i, j, k, ψ) + ε)^ƞ
@inline αy₁(i, j, k, ψ) = C3₁ / (βy₁(i, j, k, ψ) + ε)^ƞ
@inline αy₂(i, j, k, ψ) = C3₂ / (βy₂(i, j, k, ψ) + ε)^ƞ

@inline αz₀(i, j, k, ψ) = C3₀ / (βz₀(i, j, k, ψ) + ε)^ƞ
@inline αz₁(i, j, k, ψ) = C3₁ / (βz₁(i, j, k, ψ) + ε)^ƞ
@inline αz₂(i, j, k, ψ) = C3₂ / (βz₂(i, j, k, ψ) + ε)^ƞ

#####
##### WENO-5 normalized weights
#####

@inline function weno5_weights_x(i, j, k, ψ)
    α₀ = αx₀(i, j, k, ψ)
    α₁ = αx₁(i, j, k, ψ)
    α₂ = αx₂(i, j, k, ψ)
    
    Σα = α₀ + α₁ + α₂ 
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    
    return w₀, w₁, w₂
end

@inline function weno5_weights_y(i, j, k, ψ)
    α₀ = αy₀(i, j, k, ψ)
    α₁ = αy₁(i, j, k, ψ)
    α₂ = αy₂(i, j, k, ψ)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    
    return w₀, w₁, w₂
end

@inline function weno5_weights_z(i, j, k, ψ)
    α₀ = αz₀(i, j, k, ψ)
    α₁ = αz₁(i, j, k, ψ)
    α₂ = αz₂(i, j, k, ψ)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    
    return w₀, w₁, w₂
end

#####
##### WENO-5 reconstruction
#####

@inline function weno5_reconstructᶠᵃᵃ(i, j, k, grid, ψ)
    w₀, w₁, w₂ = weno5_weights_x(i, j, k, ψ)
    return w₀ * px₀(i, j, k, ψ) + w₁ * px₁(i, j, k, ψ) + w₂ * px₂(i, j, k, ψ)
end

@inline function weno5_reconstructᵃᶠᵃ(i, j, k, grid, ψ)
    w₀, w₁, w₂ = weno5_weights_y(i, j, k, ψ)
    return w₀ * py₀(i, j, k, ψ) + w₁ * py₁(i, j, k, ψ) + w₂ * py₂(i, j, k, ψ)
end

@inline function weno5_reconstructᵃᵃᶠ(i, j, k, grid, ψ)
    w₀, w₁, w₂ = weno5_weights_z(i, j, k, ψ)
    return w₀ * pz₀(i, j, k, ψ) + w₁ * pz₁(i, j, k, ψ) + w₂ * pz₂(i, j, k, ψ)
end

@inline weno5_reconstructᶜᵃᵃ(i, j, k, grid, ψ) = weno5_reconstructᶠᵃᵃ(i+1, j, k, grid, ψ)
@inline weno5_reconstructᵃᶜᵃ(i, j, k, grid, ψ) = weno5_reconstructᵃᶠᵃ(i, j+1, k, grid, ψ)
@inline weno5_reconstructᵃᵃᶜ(i, j, k, grid, ψ) = weno5_reconstructᵃᵃᶠ(i, j, k+1, grid, ψ)

# Periodic directions

@inline interpolateᶠᵃᵃ(i, j, k, grid, ::WENO5, ψ) = weno5_reconstructᶠᵃᵃ(i, j, k, grid, ψ)
@inline interpolateᵃᶠᵃ(i, j, k, grid, ::WENO5, ψ) = weno5_reconstructᵃᶠᵃ(i, j, k, grid, ψ)
@inline interpolateᵃᵃᶠ(i, j, k, grid, ::WENO5, ψ) = weno5_reconstructᵃᵃᶠ(i, j, k, grid, ψ)

@inline interpolateᶜᵃᵃ(i, j, k, grid, ::WENO5, ψ) = weno5_reconstructᶜᵃᵃ(i, j, k, grid, ψ)
@inline interpolateᵃᶜᵃ(i, j, k, grid, ::WENO5, ψ) = weno5_reconstructᵃᶜᵃ(i, j, k, grid, ψ)
@inline interpolateᵃᵃᶜ(i, j, k, grid, ::WENO5, ψ) = weno5_reconstructᵃᵃᶜ(i, j, k, grid, ψ)

# Bounded directions

@inline function interpolateᶠᵃᵃ(i, j, k, grid::AbstractGrid{FT, <:Bounded}, ::WENO5, ψ) where FT
    if i > 2 && i < grid.Nx - 2 && j > 2 && j < grid.Ny - 2 && k > 2 && k < grid.Nz - 2
        return weno5_reconstructᶠᵃᵃ(i, j, k, grid, ψ)
    else
        return ℑxᶠᵃᵃ(i, j, k, grid, ψ)
    end
end

@inline function interpolateᵃᶠᵃ(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, ::WENO5, ψ) where {FT, TX}
    if i > 2 && i < grid.Nx - 2 && j > 2 && j < grid.Ny - 2 && k > 2 && k < grid.Nz - 2
        return weno5_reconstructᵃᶠᵃ(i, j, k, grid, ψ)
    else
        return ℑyᵃᶠᵃ(i, j, k, grid, ψ)
    end
end

@inline function interpolateᵃᵃᶠ(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, ::WENO5, ψ) where {FT, TX, TY}
    if i > 2 && i < grid.Nx - 2 && j > 2 && j < grid.Ny - 2 && k > 2 && k < grid.Nz - 2
        return weno5_reconstructᵃᵃᶠ(i, j, k, grid, ψ)
    else
        return ℑzᵃᵃᶠ(i, j, k, grid, ψ)
    end
end

@inline function interpolateᶜᵃᵃ(i, j, k, grid::AbstractGrid{FT, <:Bounded}, ::WENO5, ψ) where FT
    if i > 2 && i < grid.Nx - 2 && j > 2 && j < grid.Ny - 2 && k > 2 && k < grid.Nz - 2
        return weno5_reconstructᶜᵃᵃ(i, j, k, grid, ψ)
    else
        return ℑxᶜᵃᵃ(i, j, k, grid, ψ)
    end
end

@inline function interpolateᵃᶜᵃ(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, ::WENO5, ψ) where {FT, TX}
    if i > 2 && i < grid.Nx - 2 && j > 2 && j < grid.Ny - 2 && k > 2 && k < grid.Nz - 2
        return weno5_reconstructᵃᶜᵃ(i, j, k, grid, ψ)
    else
        return ℑyᵃᶜᵃ(i, j, k, grid, ψ)
    end
end

@inline function interpolateᵃᵃᶜ(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, ::WENO5, ψ) where {FT, TX, TY}
    if i > 2 && i < grid.Nx - 2 && j > 2 && j < grid.Ny - 2 && k > 2 && k < grid.Nz - 2
        return weno5_reconstructᵃᵃᶜ(i, j, k, grid, ψ)
    else
        return ℑzᵃᵃᶜ(i, j, k, grid, ψ)
    end
end

#####
##### Momentum advection fluxes
#####

@inline momentum_flux_uu(i, j, k, grid, weno::WENO5, u)    = Axᵃᵃᶠ(i, j, k, grid) * interpolateᶜᵃᵃ(i, j, k, grid, weno, u) * interpolateᶜᵃᵃ(i, j, k, grid, weno, u)
@inline momentum_flux_uv(i, j, k, grid, weno::WENO5, u, v) = Ayᵃᵃᶠ(i, j, k, grid) * interpolateᶠᵃᵃ(i, j, k, grid, weno, v) * interpolateᵃᶠᵃ(i, j, k, grid, weno, u)
@inline momentum_flux_uw(i, j, k, grid, weno::WENO5, u, w) = Azᵃᵃᵃ(i, j, k, grid) * interpolateᶠᵃᵃ(i, j, k, grid, weno, w) * interpolateᵃᵃᶠ(i, j, k, grid, weno, u)

@inline momentum_flux_vu(i, j, k, grid, weno::WENO5, u, v) = Axᵃᵃᶠ(i, j, k, grid) * interpolateᵃᶠᵃ(i, j, k, grid, weno, u) * interpolateᶠᵃᵃ(i, j, k, grid, weno, v)
@inline momentum_flux_vv(i, j, k, grid, weno::WENO5, v)    = Ayᵃᵃᶠ(i, j, k, grid) * interpolateᵃᶜᵃ(i, j, k, grid, weno, v) * interpolateᵃᶜᵃ(i, j, k, grid, weno, v)
@inline momentum_flux_vw(i, j, k, grid, weno::WENO5, v, w) = Azᵃᵃᵃ(i, j, k, grid) * interpolateᵃᶠᵃ(i, j, k, grid, weno, w) * interpolateᵃᵃᶠ(i, j, k, grid, weno, v)

@inline momentum_flux_wu(i, j, k, grid, weno::WENO5, u, w) = Axᵃᵃᶠ(i, j, k, grid) * interpolateᵃᵃᶠ(i, j, k, grid, weno, u) * interpolateᶠᵃᵃ(i, j, k, grid, weno, w)
@inline momentum_flux_wv(i, j, k, grid, weno::WENO5, v, w) = Ayᵃᵃᶠ(i, j, k, grid) * interpolateᵃᵃᶠ(i, j, k, grid, weno, v) * interpolateᵃᶠᵃ(i, j, k, grid, weno, w)
@inline momentum_flux_ww(i, j, k, grid, weno::WENO5, w)    = Azᵃᵃᵃ(i, j, k, grid) * interpolateᵃᵃᶜ(i, j, k, grid, weno, w) * interpolateᵃᵃᶜ(i, j, k, grid, weno, w)

#####
##### Advective tracer fluxes
#####

@inline advective_tracer_flux_x(i, j, k, grid, weno::WENO5, u, c) = Ax_ψᵃᵃᶠ(i, j, k, grid, u) * interpolateᶠᵃᵃ(i, j, k, grid, weno, c)
@inline advective_tracer_flux_y(i, j, k, grid, weno::WENO5, v, c) = Ay_ψᵃᵃᶠ(i, j, k, grid, v) * interpolateᵃᶠᵃ(i, j, k, grid, weno, c)
@inline advective_tracer_flux_z(i, j, k, grid, weno::WENO5, w, c) = Az_ψᵃᵃᵃ(i, j, k, grid, w) * interpolateᵃᵃᶠ(i, j, k, grid, weno, c)
