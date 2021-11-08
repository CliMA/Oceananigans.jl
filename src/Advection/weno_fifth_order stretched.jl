#####
##### Weighted Essentially Non-Oscillatory (WENO) scheme of order 5
#####

struct WENO5S <: AbstractUpwindBiasedAdvectionScheme{2} end

@inline boundary_buffer(::WENO5S) = 2

const AG = AbstractGrid

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5S, c) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5S, c) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5S, c) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, centered_fourth_order, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::WENO5S, u) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, centered_fourth_order, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::WENO5S, v) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, centered_fourth_order, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::WENO5S, w) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, centered_fourth_order, w)

#####
##### ENO interpolants of size 3
#####

@inline left_biased_p₀(grid::AG{FT}, ψ) where FT = @inbounds + FT(1/3) * ψ[3] + FT(5/6) * ψ[4] -  FT(1/6) * ψ[5]
@inline left_biased_p₁(grid::AG{FT}, ψ) where FT = @inbounds - FT(1/6) * ψ[2] + FT(5/6) * ψ[3] +  FT(1/3) * ψ[4]
@inline left_biased_p₂(grid::AG{FT}, ψ) where FT = @inbounds + FT(1/3) * ψ[1] - FT(7/6) * ψ[2] + FT(11/6) * ψ[3]

@inline right_biased_px₀(grid::AG{FT}, ψ) where FT = @inbounds + FT(11/6) * ψ[3] - FT(7/6) * ψ[4] + FT(1/3) * ψ[5]
@inline right_biased_px₁(grid::AG{FT}, ψ) where FT = @inbounds +  FT(1/3) * ψ[2] + FT(5/6) * ψ[3] - FT(1/6) * ψ[4]
@inline right_biased_px₂(grid::AG{FT}, ψ) where FT = @inbounds -  FT(1/6) * ψ[1] + FT(5/6) * ψ[2] + FT(1/3) * ψ[3]

#####
##### Jiang & Shu (1996) WENO smoothness indicators. See also Equation 2.63 in Shu (1998).
#####

# We use 32-bit integer to represent the exponent "2" for fast exponentiation.
# See https:/github.com/CliMA/Oceananigans.jl/pull/1770 for more information.
const two_32 = Int32(2)

@inline left_biased_β₀(grid::AG{FT}, ψ) where FT = @inbounds FT(13/12) * (ψ[3] - 2ψ[4] + ψ[5])^two_32 + FT(1/4) * (3ψ[3] - 4ψ[4] +  ψ[5])^two_32
@inline left_biased_β₁(grid::AG{FT}, ψ) where FT = @inbounds FT(13/12) * (ψ[2] - 2ψ[3] + ψ[4])^two_32 + FT(1/4) * ( ψ[2]         -  ψ[4])^two_32
@inline left_biased_β₂(grid::AG{FT}, ψ) where FT = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32

# Right-biased smoothness indicators are a reflection or "symmetric modification" of the left-biased smoothness
# indicators around grid point `i-1/2`.

@inline right_biased_βx₀(grid::AG{FT}, ψ) where FT = @inbounds FT(13/12) * (ψ[3] - 2ψ[4] + ψ[5])^two_32 + FT(1/4) * ( ψ[3] - 4ψ[4] + 3ψ[5])^two_32
@inline right_biased_βx₁(grid::AG{FT}, ψ) where FT = @inbounds FT(13/12) * (ψ[2] - 2ψ[3] + ψ[4])^two_32 + FT(1/4) * ( ψ[2]         -  ψ[4])^two_32
@inline right_biased_βx₂(grid::AG{FT}, ψ) where FT = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32

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
# see https:/github.com/CliMA/Oceananigans.jl/pull/1770 for more information.
const ƞ = Int32(2) # WENO exponent
const ε = 1e-6

@inline left_biased_α₀(grid::AG{FT}, ψ) where FT = FT(C3₀) / (left_biased_β₀(grid, ψ) + FT(ε))^ƞ
@inline left_biased_α₁(grid::AG{FT}, ψ) where FT = FT(C3₁) / (left_biased_β₁(grid, ψ) + FT(ε))^ƞ
@inline left_biased_α₂(grid::AG{FT}, ψ) where FT = FT(C3₂) / (left_biased_β₂(grid, ψ) + FT(ε))^ƞ

@inline right_biased_α₀(grid::AG{FT}, ψ) where FT = FT(C3₂) / (right_biased_β₀(grid, ψ) + FT(ε))^ƞ
@inline right_biased_α₁(grid::AG{FT}, ψ) where FT = FT(C3₁) / (right_biased_β₁(grid, ψ) + FT(ε))^ƞ
@inline right_biased_α₂(grid::AG{FT}, ψ) where FT = FT(C3₀) / (right_biased_β₂(grid, ψ) + FT(ε))^ƞ

#####
##### WENO-5 normalized weights
#####

@inline function left_biased_weno5_weights(grid, ψ)
    α₀ = left_biased_α₀(grid, ψ)
    α₁ = left_biased_α₁(grid, ψ)
    α₂ = left_biased_α₂(grid, ψ)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function right_biased_weno5_weights(grid, ψ)

    α₀ = right_biased_α₀(grid, ψ)
    α₁ = right_biased_α₁(grid, ψ)
    α₂ = right_biased_α₂(grid, ψ)

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
    ψₛ = calc_stencil_left_x(grid, ψ, i, j, k)
    w₀, w₁, w₂ = left_biased_weno5_weights(grid, ψₛ)
    return w₀ * left_biased_p₀(grid, ψₛ) + w₁ * left_biased_p₁(grid, ψₛ) + w₂ * left_biased_p₂(grid, ψₛ)
end

@inline function left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5, ψ)
    ψₛ = calc_stencil_left_y(grid, ψ, i, j, k)
    w₀, w₁, w₂ = left_biased_weno5_weights(grid, ψₛ)
    return w₀ * left_biased_p₀(grid, ψₛ) + w₁ * left_biased_p₁(grid, ψₛ) + w₂ * left_biased_p₂(grid, ψₛ)
end

@inline function left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5, ψ)
    ψₛ = calc_stencil_left_z(grid, ψ, i, j, k)
    w₀, w₁, w₂ = left_biased_weno5_weights(grid, ψₛ)
    return w₀ * left_biased_p₀(grid, ψₛ) + w₁ * left_biased_p₁(grid, ψₛ) + w₂ * left_biased_p₂(grid, ψₛ)
end

@inline function right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5, ψ)
    ψₛ = calc_stencil_right_x(grid, ψ, i, j, k)
    w₀, w₁, w₂ = right_biased_weno5_weights(grid, ψₛ)
    return w₀ * right_biased_p₀(grid, ψₛ) + w₁ * right_biased_p₁(grid, ψₛ) + w₂ * right_biased_p₂(grid, ψₛ)
end

@inline function right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5, ψ)
    ψₛ = calc_stencil_right_y(grid, ψ, i, j, k)
    w₀, w₁, w₂ = right_biased_weno5_weights(grid, ψₛ)
    return w₀ * right_biased_p₀(grid, ψₛ) + w₁ * right_biased_p₁(grid, ψₛ) + w₂ * right_biased_p₂(grid, ψₛ)
end

@inline function right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5, ψ)
    ψₛ = calc_stencil_right_z(grid, ψ, i, j, k)
    w₀, w₁, w₂ = right_biased_weno5_weights(grid, ψₛ)
    return w₀ * right_biased_p₀(grid, ψₛ) + w₁ * right_biased_p₁(grid, ψₛ) + w₂ * right_biased_p₂(grid, ψₛ)
end

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5, ψ) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5, ψ) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5, ψ) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5, ψ) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5, ψ) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5, ψ) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ)

function calc_stencil_left_x(grid, ψ, i, j, k)

    ψₛ = zeros(eltype(grid), 5)
    
    for h = -3:1
        ψₛ[h+4] = 
    end


end

@inline calc_stencil_left_x(grid::XRegRectilinearGrid, ψ, i, j, k) = @view [ψ[i-3, j, k], ψ[i-2, j, k], ψ[i-1, j, k], ψ[i, j, k], ψ[i+1, j, k]]
@inline calc_stencil_left_y(grid::YRegRectilinearGrid, ψ, i, j, k) = @view [ψ[i, j-3, k], ψ[i, j-2, k], ψ[i, j-1, k], ψ[i, j, k], ψ[i, j+1, k]]
@inline calc_stencil_left_z(grid::ZRegRectilinearGrid, ψ, i, j, k) = @view [ψ[i, j, k-3], ψ[i, j, k-2], ψ[i, j, k-1], ψ[i, j, k], ψ[i, j, k+1]]

@inline calc_stencil_right_x(grid::XRegRectilinearGrid, ψ, i, j, k) = @view [ψ[i-2, j, k], ψ[i-1, j, k], ψ[i, j, k], ψ[i+1, j, k], ψ[i+2, j, k]]
@inline calc_stencil_right_x(grid::YRegRectilinearGrid, ψ, i, j, k) = @view [ψ[i, j-2, k], ψ[i, j-1, k], ψ[i, j, k], ψ[i, j+1, k], ψ[i, j+2, k]]
@inline calc_stencil_right_x(grid::ZRegRectilinearGrid, ψ, i, j, k) = @view [ψ[i, j, k-2], ψ[i, j, k-1], ψ[i, j, k], ψ[i, j, k+1], ψ[i, j, k+2]]
