#####
##### Weighted Essentially Non-Oscillatory (WENO) scheme of order 5
#####

struct WENO5 <: AbstractAdvectionScheme end

#####
##### ENO interpolants of size 3
#####

@inline px₀(i, j, k, f) = @inbounds + 1/3 * f[i-1, j, k] + 5/6 * f[i,   j, k] -  1/6 * f[i+1, j, k]
@inline px₁(i, j, k, f) = @inbounds - 1/6 * f[i-2, j, k] + 5/6 * f[i-1, j, k] +  1/3 * f[i,   j, k]
@inline px₂(i, j, k, f) = @inbounds + 1/3 * f[i-3, j, k] - 7/6 * f[i-2, j, k] + 11/6 * f[i-1, j, k]

@inline py₀(i, j, k, f) = @inbounds + 1/3 * f[i, j-1, k] + 5/6 * f[i, j,   k] -  1/6 * f[i, j+1, k]
@inline py₁(i, j, k, f) = @inbounds - 1/6 * f[i, j-2, k] + 5/6 * f[i, j-1, k] +  1/3 * f[i, j  , k]
@inline py₂(i, j, k, f) = @inbounds + 1/3 * f[i, j-3, k] - 7/6 * f[i, j-2, k] + 11/6 * f[i, j-1, k]

@inline pz₀(i, j, k, f) = @inbounds + 1/3 * f[i, j, k-1] + 5/6 * f[i, j,   k] -  1/6 * f[i, j, k+1]
@inline pz₁(i, j, k, f) = @inbounds - 1/6 * f[i, j, k-2] + 5/6 * f[i, j, k-1] +  1/3 * f[i, j,   k]
@inline pz₂(i, j, k, f) = @inbounds + 1/3 * f[i, j, k-3] - 7/6 * f[i, j, k-2] + 11/6 * f[i, j, k-1]

#####
##### Jiang & Shu (1996) WENO smoothness indicators
#####

@inline βx₀(i, j, k, f) = @inbounds 13/12 * (f[i-1, j, k] - 2f[i,   j, k] + f[i+1, j, k])^2 + 1/4 * (3f[i-1, j, k] - 4f[i,   j, k] +  f[i+1, j, k])^2
@inline βx₁(i, j, k, f) = @inbounds 13/12 * (f[i-2, j, k] - 2f[i-1, j, k] + f[i,   j, k])^2 + 1/4 * ( f[i-2, j, k]                 -  f[i,   j, k])^2
@inline βx₂(i, j, k, f) = @inbounds 13/12 * (f[i-3, j, k] - 2f[i-2, j, k] + f[i-1, j, k])^2 + 1/4 * ( f[i-3, j, k] - 4f[i-2, j, k] + 3f[i-1, j, k])^2

@inline βy₀(i, j, k, f) = @inbounds 13/12 * (f[i, j-1, k] - 2f[i, j,   k] + f[i, j+1, k])^2 + 1/4 * (3f[i, j-1, k] - 4f[i,   j, k] +  f[i, j+1, k])^2
@inline βy₁(i, j, k, f) = @inbounds 13/12 * (f[i, j-2, k] - 2f[i, j-1, k] + f[i, j,   k])^2 + 1/4 * ( f[i, j-2, k]                 -  f[i,   j, k])^2
@inline βy₂(i, j, k, f) = @inbounds 13/12 * (f[i, j-3, k] - 2f[i, j-2, k] + f[i, j-1, k])^2 + 1/4 * ( f[i, j-3, k] - 4f[i, j-2, k] + 3f[i, j-1, k])^2

@inline βz₀(i, j, k, f) = @inbounds 13/12 * (f[i, j, k-1] - 2f[i, j,   k] + f[i, j, k+1])^2 + 1/4 * (3f[i, j, k-1] - 4f[i, j,   k] +  f[i, j, k+1])^2
@inline βz₁(i, j, k, f) = @inbounds 13/12 * (f[i, j, k-2] - 2f[i, j, k-1] + f[i, j,   k])^2 + 1/4 * ( f[i, j, k-2]                 -  f[i, j,   k])^2
@inline βz₂(i, j, k, f) = @inbounds 13/12 * (f[i, j, k-3] - 2f[i, j, k-2] + f[i, j, k-1])^2 + 1/4 * ( f[i, j, k-3] - 4f[i, j, k-2] + 3f[i, j, k-1])^2

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

@inline αx₀(i, j, k, f) = C3₀ / (βx₀(i, j, k, f) + ε)^ƞ
@inline αx₁(i, j, k, f) = C3₁ / (βx₁(i, j, k, f) + ε)^ƞ
@inline αx₂(i, j, k, f) = C3₂ / (βx₂(i, j, k, f) + ε)^ƞ

@inline αy₀(i, j, k, f) = C3₀ / (βy₀(i, j, k, f) + ε)^ƞ
@inline αy₁(i, j, k, f) = C3₁ / (βy₁(i, j, k, f) + ε)^ƞ
@inline αy₂(i, j, k, f) = C3₂ / (βy₂(i, j, k, f) + ε)^ƞ

@inline αz₀(i, j, k, f) = C3₀ / (βz₀(i, j, k, f) + ε)^ƞ
@inline αz₁(i, j, k, f) = C3₁ / (βz₁(i, j, k, f) + ε)^ƞ
@inline αz₂(i, j, k, f) = C3₂ / (βz₂(i, j, k, f) + ε)^ƞ

#####
##### WENO-5 normalized weights
#####

@inline function weno5_weights_x(i, j, k, f)
    α₀ = αx₀(i, j, k, f)
    α₁ = αx₁(i, j, k, f)
    α₂ = αx₂(i, j, k, f)
    
    Σα = α₀ + α₁ + α₂ 
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    
    return w₀, w₁, w₂
end

@inline function weno5_weights_y(i, j, k, f)
    α₀ = αy₀(i, j, k, f)
    α₁ = αy₁(i, j, k, f)
    α₂ = αy₂(i, j, k, f)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    
    return w₀, w₁, w₂
end

@inline function weno5_weights_z(i, j, k, f)
    α₀ = αz₀(i, j, k, f)
    α₁ = αz₁(i, j, k, f)
    α₂ = αz₂(i, j, k, f)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα
    
    return w₀, w₁, w₂
end

#####
##### WENO-5 flux reconstruction
#####

@inline function weno5_flux_x(i, j, k, f)
    w₀, w₁, w₂ = weno5_weights_x(i, j, k, f)
    return w₀ * px₀(i, j, k, f) + w₁ * px₁(i, j, k, f) + w₂ * px₂(i, j, k, f)
end

@inline function weno5_flux_y(i, j, k, f)
    w₀, w₁, w₂ = weno5_weights_y(i, j, k, f)
    return w₀ * py₀(i, j, k, f) + w₁ * py₁(i, j, k, f) + w₂ * py₂(i, j, k, f)
end

@inline function weno5_flux_z(i, j, k, f)
    w₀, w₁, w₂ = weno5_weights_z(i, j, k, f)
    return w₀ * pz₀(i, j, k, f) + w₁ * pz₁(i, j, k, f) + w₂ * pz₂(i, j, k, f)
end

#####
##### Momentum advection fluxes
#####

@inline momentum_flux_uu_weno5(i, j, k, grid, u)    = ℑxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno5_flux_x(i+1, j, k, u)
@inline momentum_flux_uv_weno5(i, j, k, grid, u, v) = ℑxᶠᵃᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno5_flux_y(i,   j, k, u)
@inline momentum_flux_uw_weno5(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno5_flux_z(i,   j, k, u)

@inline momentum_flux_vu_weno5(i, j, k, grid, u, v) = ℑyᵃᶠᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno5_flux_x(i, j,   k, v)
@inline momentum_flux_vv_weno5(i, j, k, grid, v)    = ℑyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno5_flux_y(i, j+1, k, v)
@inline momentum_flux_vw_weno5(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno5_flux_z(i, j,   k, v)

@inline momentum_flux_wu_weno5(i, j, k, grid, u, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * weno5_flux_x(i, j, k,   w)
@inline momentum_flux_wv_weno5(i, j, k, grid, v, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * weno5_flux_y(i, j, k,   w)
@inline momentum_flux_ww_weno5(i, j, k, grid, w)    = ℑzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * weno5_flux_z(i, j, k+1, w)

# Periodic directions

@inline momentum_flux_uu(i, j, k, grid, ::WENO5, u)    = momentum_flux_uu_weno5(i, j, k, grid, u)
@inline momentum_flux_uv(i, j, k, grid, ::WENO5, u, v) = momentum_flux_uv_weno5(i, j, k, grid, u, v)
@inline momentum_flux_uw(i, j, k, grid, ::WENO5, u, w) = momentum_flux_uw_weno5(i, j, k, grid, u, w)

@inline momentum_flux_vu(i, j, k, grid, ::WENO5, u, v) = momentum_flux_vu_weno5(i, j, k, grid, u, v)
@inline momentum_flux_vv(i, j, k, grid, ::WENO5, v)    = momentum_flux_vv_weno5(i, j, k, grid, v)
@inline momentum_flux_vw(i, j, k, grid, ::WENO5, v, w) = momentum_flux_vw_weno5(i, j, k, grid, v, w)

@inline momentum_flux_wu(i, j, k, grid, ::WENO5, u, w) = momentum_flux_wu_weno5(i, j, k, grid, w, u)
@inline momentum_flux_wv(i, j, k, grid, ::WENO5, v, w) = momentum_flux_wv_weno5(i, j, k, grid, w, v)
@inline momentum_flux_ww(i, j, k, grid, ::WENO5, w)    = momentum_flux_ww_weno5(i, j, k, grid, w)

# Bounded directions

@inline function momentum_flux_uu(i, j, k, grid::AbstractGrid{FT, <:Bounded}, ::WENO5, u) where FT
    if i > 2 && i < grid.Nx - 1
        return momentum_flux_uu_weno5(i, j, k, grid, u)
    else
        return momentum_flux_uu(i, j, k, grid, centered_second_order, u)
    end
end

@inline function momentum_flux_uv(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, ::WENO5, u, v) where {FT, TX}
    if j > 2 && j < grid.Ny - 1
        return momentum_flux_uv_weno5(i, j, k, grid, u, v)
    else
        return momentum_flux_uv(i, j, k, grid, centered_second_order, u, v)
    end
end

@inline function momentum_flux_uw(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, ::WENO5, u, w) where {FT, TX, TY}
    if k > 2 && k < grid.Nz - 1
        return momentum_flux_uw_weno5(i, j, k, grid, u, w)
    else
        return momentum_flux_uw(i, j, k, grid, centered_second_order, u, w)
    end
end

@inline function momentum_flux_vu(i, j, k, grid::AbstractGrid{FT, <:Bounded}, ::WENO5, u, v) where {FT}
    if i > 2 && i < grid.Nx - 1
        return momentum_flux_vu_weno5(i, j, k, grid, u, v)
    else
        return momentum_flux_vu(i, j, k, grid, centered_second_order, u, v)
    end
end

@inline function momentum_flux_vv(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, ::WENO5, v) where {FT, TX}
    if j > 2 && j < grid.Ny - 1
        return momentum_flux_vv_weno5(i, j, k, grid, v)
    else
        return momentum_flux_vv(i, j, k, grid, centered_second_order, v)
    end
end

@inline function momentum_flux_vw(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, ::WENO5, v, w) where {FT, TX, TY}
    if k > 2 && k < grid.Nz - 1
        return momentum_flux_vw_weno5(i, j, k, grid, v, w)
    else
        return momentum_flux_vw(i, j, k, grid, centered_second_order, v, w)
    end
end

@inline function momentum_flux_wu(i, j, k, grid::AbstractGrid{FT, <:Bounded}, ::WENO5, u, w) where FT
    if i > 2 && i < grid.Nx - 1
        return momentum_flux_wu_weno5(i, j, k, grid, u, w)
    else
        return momentum_flux_wu(i, j, k, grid, centered_second_order, u, w)
    end
end

@inline function momentum_flux_wv(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, ::WENO5, v, w) where {FT, TX}
    if j > 2 && j < grid.Ny - 1
        return momentum_flux_wv_weno5(i, j, k, grid, v, w)
    else
        return momentum_flux_wv(i, j, k, grid, centered_second_order, v, w)
    end
end

@inline function momentum_flux_ww(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, ::WENO5, w) where {FT, TX, TY}
    if k > 2 && k < grid.Nz - 1
        return momentum_flux_ww_weno5(i, j, k, grid, w)
    else
        return momentum_flux_ww(i, j, k, grid, centered_second_order, w)
    end
end

#####
##### Advective tracer fluxes
#####

@inline advective_tracer_flux_x_weno5(i, j, k, grid, u, c) = Ax_ψᵃᵃᶠ(i, j, k, grid, u) * weno5_flux_x(i, j, k, c)
@inline advective_tracer_flux_y_weno5(i, j, k, grid, v, c) = Ay_ψᵃᵃᶠ(i, j, k, grid, v) * weno5_flux_y(i, j, k, c)
@inline advective_tracer_flux_z_weno5(i, j, k, grid, w, c) = Az_ψᵃᵃᵃ(i, j, k, grid, w) * weno5_flux_z(i, j, k, c)

@inline advective_tracer_flux_x(i, j, k, grid, ::WENO5, u, c) = advective_tracer_flux_x_weno5(i, j, k, grid, u, c)
@inline advective_tracer_flux_y(i, j, k, grid, ::WENO5, v, c) = advective_tracer_flux_y_weno5(i, j, k, grid, v, c)
@inline advective_tracer_flux_z(i, j, k, grid, ::WENO5, w, c) = advective_tracer_flux_z_weno5(i, j, k, grid, w, c)

@inline function advective_tracer_flux_x(i, j, k, grid::AbstractGrid{FT, <:Bounded}, ::WENO5, u, c) where FT
    if i > 2 && i < grid.Nx - 1
        return advective_tracer_flux_x_weno5(i, j, k, grid, u, c)
    else
        return advective_tracer_flux_x(i, j, k, grid, centered_second_order, u, c)
    end
end

@inline function advective_tracer_flux_y(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, ::WENO5, v, c) where {FT, TX}
    if j > 2 && j < grid.Ny - 1
        return advective_tracer_flux_y_weno5(i, j, k, grid, v, c)
    else
        return advective_tracer_flux_y(i, j, k, grid, centered_second_order, v, c)
    end
end

@inline function advective_tracer_flux_z(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, ::WENO5, w, c) where {FT, TX, TY}
    if k > 2 && k < grid.Nz - 1
        return advective_tracer_flux_z_weno5(i, j, k, grid, w, c)
    else
        return advective_tracer_flux_z(i, j, k, grid, centered_second_order, w, c)
    end
end
