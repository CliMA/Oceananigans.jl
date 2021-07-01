#####
##### Weighted Essentially Non-Oscillatory (WENO) scheme of order 5
#####

struct WENO5 <: AbstractUpwindBiasedAdvectionScheme{2} end

@inline boundary_buffer(::WENO5) = 2

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5, c) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5, c) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5, c) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, centered_fourth_order, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::WENO5, u) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, centered_fourth_order, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::WENO5, v) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, centered_fourth_order, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::WENO5, w) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, centered_fourth_order, w)

#####
##### ENO interpolants of size 3
#####

@inline left_biased_px₀(i, j, k, ψ) = @inbounds + 1/3 * ψ[i-1, j, k] + 5/6 * ψ[i,   j, k] -  1/6 * ψ[i+1, j, k]
@inline left_biased_px₁(i, j, k, ψ) = @inbounds - 1/6 * ψ[i-2, j, k] + 5/6 * ψ[i-1, j, k] +  1/3 * ψ[i,   j, k]
@inline left_biased_px₂(i, j, k, ψ) = @inbounds + 1/3 * ψ[i-3, j, k] - 7/6 * ψ[i-2, j, k] + 11/6 * ψ[i-1, j, k]

@inline left_biased_py₀(i, j, k, ψ) = @inbounds + 1/3 * ψ[i, j-1, k] + 5/6 * ψ[i, j,   k] -  1/6 * ψ[i, j+1, k]
@inline left_biased_py₁(i, j, k, ψ) = @inbounds - 1/6 * ψ[i, j-2, k] + 5/6 * ψ[i, j-1, k] +  1/3 * ψ[i, j  , k]
@inline left_biased_py₂(i, j, k, ψ) = @inbounds + 1/3 * ψ[i, j-3, k] - 7/6 * ψ[i, j-2, k] + 11/6 * ψ[i, j-1, k]

@inline left_biased_pz₀(i, j, k, ψ) = @inbounds + 1/3 * ψ[i, j, k-1] + 5/6 * ψ[i, j,   k] -  1/6 * ψ[i, j, k+1]
@inline left_biased_pz₁(i, j, k, ψ) = @inbounds - 1/6 * ψ[i, j, k-2] + 5/6 * ψ[i, j, k-1] +  1/3 * ψ[i, j,   k]
@inline left_biased_pz₂(i, j, k, ψ) = @inbounds + 1/3 * ψ[i, j, k-3] - 7/6 * ψ[i, j, k-2] + 11/6 * ψ[i, j, k-1]

@inline right_biased_px₀(i, j, k, ψ) = @inbounds + 11/6 * ψ[i,   j, k] - 7/6 * ψ[i+1, j, k] + 1/3 * ψ[i+2, j, k]
@inline right_biased_px₁(i, j, k, ψ) = @inbounds +  1/3 * ψ[i-1, j, k] + 5/6 * ψ[i,   j, k] - 1/6 * ψ[i+1, j, k]
@inline right_biased_px₂(i, j, k, ψ) = @inbounds -  1/6 * ψ[i-2, j, k] + 5/6 * ψ[i-1, j, k] + 1/3 * ψ[i,   j, k]

@inline right_biased_py₀(i, j, k, ψ) = @inbounds + 11/6 * ψ[i,   j, k] - 7/6 * ψ[i, j+1, k] + 1/3 * ψ[i, j+2, k]
@inline right_biased_py₁(i, j, k, ψ) = @inbounds +  1/3 * ψ[i, j-1, k] + 5/6 * ψ[i,   j, k] - 1/6 * ψ[i, j+1, k]
@inline right_biased_py₂(i, j, k, ψ) = @inbounds -  1/6 * ψ[i, j-2, k] + 5/6 * ψ[i, j-1, k] + 1/3 * ψ[i,   j, k]

@inline right_biased_pz₀(i, j, k, ψ) = @inbounds + 11/6 * ψ[i, j,   k] - 7/6 * ψ[i, j, k+1] + 1/3 * ψ[i, j, k+2]
@inline right_biased_pz₁(i, j, k, ψ) = @inbounds +  1/3 * ψ[i, j, k-1] + 5/6 * ψ[i, j,   k] - 1/6 * ψ[i, j, k+1]
@inline right_biased_pz₂(i, j, k, ψ) = @inbounds -  1/6 * ψ[i, j, k-2] + 5/6 * ψ[i, j, k-1] + 1/3 * ψ[i, j,   k]

#####
##### Jiang & Shu (1996) WENO smoothness indicators. See also Equation 2.63 in Shu (1998).
#####

# We use 32-bit integer to represent the exponent "2" for fast exponentiation.
# See https://github.com/CliMA/Oceananigans.jl/pull/1770 for more information.
const two_32 = Int32(2) 

@inline left_biased_βx₀(i, j, k, ψ) = @inbounds 13/12 * (ψ[i-1, j, k] - 2ψ[i,   j, k] + ψ[i+1, j, k])^two_32 + 1/4 * (3ψ[i-1, j, k] - 4ψ[i,   j, k] +  ψ[i+1, j, k])^two_32
@inline left_biased_βx₁(i, j, k, ψ) = @inbounds 13/12 * (ψ[i-2, j, k] - 2ψ[i-1, j, k] + ψ[i,   j, k])^two_32 + 1/4 * ( ψ[i-2, j, k]                 -  ψ[i,   j, k])^two_32
@inline left_biased_βx₂(i, j, k, ψ) = @inbounds 13/12 * (ψ[i-3, j, k] - 2ψ[i-2, j, k] + ψ[i-1, j, k])^two_32 + 1/4 * ( ψ[i-3, j, k] - 4ψ[i-2, j, k] + 3ψ[i-1, j, k])^two_32

@inline left_biased_βy₀(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j-1, k] - 2ψ[i, j,   k] + ψ[i, j+1, k])^two_32 + 1/4 * (3ψ[i, j-1, k] - 4ψ[i,   j, k] +  ψ[i, j+1, k])^two_32
@inline left_biased_βy₁(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j-2, k] - 2ψ[i, j-1, k] + ψ[i, j,   k])^two_32 + 1/4 * ( ψ[i, j-2, k]                 -  ψ[i,   j, k])^two_32
@inline left_biased_βy₂(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j-3, k] - 2ψ[i, j-2, k] + ψ[i, j-1, k])^two_32 + 1/4 * ( ψ[i, j-3, k] - 4ψ[i, j-2, k] + 3ψ[i, j-1, k])^two_32

@inline left_biased_βz₀(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j, k-1] - 2ψ[i, j,   k] + ψ[i, j, k+1])^two_32 + 1/4 * (3ψ[i, j, k-1] - 4ψ[i, j,   k] +  ψ[i, j, k+1])^two_32
@inline left_biased_βz₁(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j, k-2] - 2ψ[i, j, k-1] + ψ[i, j,   k])^two_32 + 1/4 * ( ψ[i, j, k-2]                 -  ψ[i, j,   k])^two_32
@inline left_biased_βz₂(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j, k-3] - 2ψ[i, j, k-2] + ψ[i, j, k-1])^two_32 + 1/4 * ( ψ[i, j, k-3] - 4ψ[i, j, k-2] + 3ψ[i, j, k-1])^two_32

# Right-biased smoothness indicators are a reflection or "symmetric modification" of the left-biased smoothness
# indicators around grid point `i-1/2`.

@inline right_biased_βx₀(i, j, k, ψ) = @inbounds 13/12 * (ψ[i,   j, k] - 2ψ[i+1, j, k] + ψ[i+2, j, k])^two_32 + 1/4 * ( ψ[i,   j, k] - 4ψ[i+1, j, k] + 3ψ[i+2, j, k])^two_32
@inline right_biased_βx₁(i, j, k, ψ) = @inbounds 13/12 * (ψ[i-1, j, k] - 2ψ[i,   j, k] + ψ[i+1, j, k])^two_32 + 1/4 * ( ψ[i-1, j, k]                 -  ψ[i+1, j, k])^two_32
@inline right_biased_βx₂(i, j, k, ψ) = @inbounds 13/12 * (ψ[i-2, j, k] - 2ψ[i-1, j, k] + ψ[i,   j, k])^two_32 + 1/4 * (3ψ[i-2, j, k] - 4ψ[i-1, j, k] +  ψ[i,   j, k])^two_32

@inline right_biased_βy₀(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j,   k] - 2ψ[i, j+1, k] + ψ[i, j+2, k])^two_32 + 1/4 * ( ψ[i,   j, k] - 4ψ[i, j+1, k] + 3ψ[i, j+2, k])^two_32
@inline right_biased_βy₁(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j-1, k] - 2ψ[i, j,   k] + ψ[i, j+1, k])^two_32 + 1/4 * ( ψ[i, j-1, k]                 -  ψ[i, j+1, k])^two_32
@inline right_biased_βy₂(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j-2, k] - 2ψ[i, j-1, k] + ψ[i, j,   k])^two_32 + 1/4 * (3ψ[i, j-2, k] - 4ψ[i, j-1, k] +  ψ[i,   j, k])^two_32

@inline right_biased_βz₀(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j,   k] - 2ψ[i, j, k+1] + ψ[i, j, k+2])^two_32 + 1/4 * ( ψ[i, j,   k] - 4ψ[i, j, k+1] + 3ψ[i, j, k+2])^two_32
@inline right_biased_βz₁(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j, k-1] - 2ψ[i, j,   k] + ψ[i, j, k+1])^two_32 + 1/4 * ( ψ[i, j, k-1]                 -  ψ[i, j, k+1])^two_32
@inline right_biased_βz₂(i, j, k, ψ) = @inbounds 13/12 * (ψ[i, j, k-2] - 2ψ[i, j, k-1] + ψ[i, j,   k])^two_32 + 1/4 * (3ψ[i, j, k-2] - 4ψ[i, j, k-1] +  ψ[i, j,   k])^two_32

#####
##### WENO-5 optimal weights
#####

const C3₀ = 3/10
const C3₁ = 3/5
const C3₂ = 1/10

#####
##### WENO-5 raw weights
#####

# Note: these constants may need to be changed for smooth solutions and/or fine grid.
# Note note: we use 32-bit integer to represent the exponent "2" for fast exponentiation.
# see https://github.com/CliMA/Oceananigans.jl/pull/1770 for more information.
const ƞ = Int32(2) # WENO exponent
const ε = 1e-6

@inline left_biased_αx₀(i, j, k, ψ) = C3₀ / (left_biased_βx₀(i, j, k, ψ) + ε)^ƞ
@inline left_biased_αx₁(i, j, k, ψ) = C3₁ / (left_biased_βx₁(i, j, k, ψ) + ε)^ƞ
@inline left_biased_αx₂(i, j, k, ψ) = C3₂ / (left_biased_βx₂(i, j, k, ψ) + ε)^ƞ

@inline left_biased_αy₀(i, j, k, ψ) = C3₀ / (left_biased_βy₀(i, j, k, ψ) + ε)^ƞ
@inline left_biased_αy₁(i, j, k, ψ) = C3₁ / (left_biased_βy₁(i, j, k, ψ) + ε)^ƞ
@inline left_biased_αy₂(i, j, k, ψ) = C3₂ / (left_biased_βy₂(i, j, k, ψ) + ε)^ƞ

@inline left_biased_αz₀(i, j, k, ψ) = C3₀ / (left_biased_βz₀(i, j, k, ψ) + ε)^ƞ
@inline left_biased_αz₁(i, j, k, ψ) = C3₁ / (left_biased_βz₁(i, j, k, ψ) + ε)^ƞ
@inline left_biased_αz₂(i, j, k, ψ) = C3₂ / (left_biased_βz₂(i, j, k, ψ) + ε)^ƞ

@inline right_biased_αx₀(i, j, k, ψ) = C3₂ / (right_biased_βx₀(i, j, k, ψ) + ε)^ƞ
@inline right_biased_αx₁(i, j, k, ψ) = C3₁ / (right_biased_βx₁(i, j, k, ψ) + ε)^ƞ
@inline right_biased_αx₂(i, j, k, ψ) = C3₀ / (right_biased_βx₂(i, j, k, ψ) + ε)^ƞ

@inline right_biased_αy₀(i, j, k, ψ) = C3₂ / (right_biased_βy₀(i, j, k, ψ) + ε)^ƞ
@inline right_biased_αy₁(i, j, k, ψ) = C3₁ / (right_biased_βy₁(i, j, k, ψ) + ε)^ƞ
@inline right_biased_αy₂(i, j, k, ψ) = C3₀ / (right_biased_βy₂(i, j, k, ψ) + ε)^ƞ

@inline right_biased_αz₀(i, j, k, ψ) = C3₂ / (right_biased_βz₀(i, j, k, ψ) + ε)^ƞ
@inline right_biased_αz₁(i, j, k, ψ) = C3₁ / (right_biased_βz₁(i, j, k, ψ) + ε)^ƞ
@inline right_biased_αz₂(i, j, k, ψ) = C3₀ / (right_biased_βz₂(i, j, k, ψ) + ε)^ƞ

#####
##### WENO-5 normalized weights
#####

@inline function left_biased_weno5_weights_x(i, j, k, ψ)
    α₀ = left_biased_αx₀(i, j, k, ψ)
    α₁ = left_biased_αx₁(i, j, k, ψ)
    α₂ = left_biased_αx₂(i, j, k, ψ)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function left_biased_weno5_weights_y(i, j, k, ψ)
    α₀ = left_biased_αy₀(i, j, k, ψ)
    α₁ = left_biased_αy₁(i, j, k, ψ)
    α₂ = left_biased_αy₂(i, j, k, ψ)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function left_biased_weno5_weights_z(i, j, k, ψ)
    α₀ = left_biased_αz₀(i, j, k, ψ)
    α₁ = left_biased_αz₁(i, j, k, ψ)
    α₂ = left_biased_αz₂(i, j, k, ψ)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function right_biased_weno5_weights_x(i, j, k, ψ)
    α₀ = right_biased_αx₀(i, j, k, ψ)
    α₁ = right_biased_αx₁(i, j, k, ψ)
    α₂ = right_biased_αx₂(i, j, k, ψ)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function right_biased_weno5_weights_y(i, j, k, ψ)
    α₀ = right_biased_αy₀(i, j, k, ψ)
    α₁ = right_biased_αy₁(i, j, k, ψ)
    α₂ = right_biased_αy₂(i, j, k, ψ)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function right_biased_weno5_weights_z(i, j, k, ψ)
    α₀ = right_biased_αz₀(i, j, k, ψ)
    α₁ = right_biased_αz₁(i, j, k, ψ)
    α₂ = right_biased_αz₂(i, j, k, ψ)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

#####
##### WENO-5 reconstruction
#####

@inline function left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5, ψ)
    w₀, w₁, w₂ = left_biased_weno5_weights_x(i, j, k, ψ)
    return w₀ * left_biased_px₀(i, j, k, ψ) + w₁ * left_biased_px₁(i, j, k, ψ) + w₂ * left_biased_px₂(i, j, k, ψ)
end

@inline function left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5, ψ)
    w₀, w₁, w₂ = left_biased_weno5_weights_y(i, j, k, ψ)
    return w₀ * left_biased_py₀(i, j, k, ψ) + w₁ * left_biased_py₁(i, j, k, ψ) + w₂ * left_biased_py₂(i, j, k, ψ)
end

@inline function left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5, ψ)
    w₀, w₁, w₂ = left_biased_weno5_weights_z(i, j, k, ψ)
    return w₀ * left_biased_pz₀(i, j, k, ψ) + w₁ * left_biased_pz₁(i, j, k, ψ) + w₂ * left_biased_pz₂(i, j, k, ψ)
end

@inline function right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5, ψ)
    w₀, w₁, w₂ = right_biased_weno5_weights_x(i, j, k, ψ)
    return w₀ * right_biased_px₀(i, j, k, ψ) + w₁ * right_biased_px₁(i, j, k, ψ) + w₂ * right_biased_px₂(i, j, k, ψ)
end

@inline function right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5, ψ)
    w₀, w₁, w₂ = right_biased_weno5_weights_y(i, j, k, ψ)
    return w₀ * right_biased_py₀(i, j, k, ψ) + w₁ * right_biased_py₁(i, j, k, ψ) + w₂ * right_biased_py₂(i, j, k, ψ)
end

@inline function right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5, ψ)
    w₀, w₁, w₂ = right_biased_weno5_weights_z(i, j, k, ψ)
    return w₀ * right_biased_pz₀(i, j, k, ψ) + w₁ * right_biased_pz₁(i, j, k, ψ) + w₂ * right_biased_pz₂(i, j, k, ψ)
end

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5, ψ) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5, ψ) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5, ψ) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5, ψ) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5, ψ) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5, ψ) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ)
