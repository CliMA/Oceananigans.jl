#####
##### Weighted Essentially Non-Oscillatory (WENO) scheme of order 5
#####

"""

GT here is used as a variable to decide wether to dispatch 
on a "regular formulation" or a "stretched formulation"

GT can take up the value of 
- Val(0) => regular weno formulation
- Val(1) => stretched weno formulation

where the two WENO schemes differ like this

u̅ᵢ = (1 - γ) uᵢ + γ uᵢ₊₁

where 

γ = Δxᵢ / (Δxᵢ + Δxᵢ₊₁) (with Δx calculated at the center of the cell)

Uniform :

p₀_left = + 1/3 * uᵢ₋₁ + 5/6 * uᵢ   -  1/6 * uᵢ₊₁
p₁_left = - 1/6 * uᵢ₋₂ + 5/6 * uᵢ₋₁ +  1/3 * uᵢ
p₂_left = + 1/3 * uᵢ₋₃ - 7/6 * uᵢ₋₂ + 11/6 * uᵢ₋₁

Stretched :

p₀_left = + 2/3 * u̅ᵢ₋₁ +  2/3 * uᵢ   -  1/3 * u̅ᵢ
p₁_left = - 1/3 * u̅ᵢ₋₂ +  4/6 * uᵢ₋₁ +  2/3 * u̅ᵢ₋₁
p₂_left = + 2/3 * u̅ᵢ₋₃ - 10/3 * uᵢ₋₂ + 11/3 * u̅ᵢ₋₂

Uniform :

p₀_right = + 11/6 * uᵢ   - 7/6 * uᵢ₊₁  + 1/3 * uᵢ₊₂  
p₁_right = +  1/3 * uᵢ₋₁ + 5/6 * uᵢ    - 1/6 * uᵢ₊₁    
p₂_right = -  1/6 * uᵢ₋₂ + 5/6 * uᵢ₋₁  + 1/3 * uᵢ  

Stretched

p₀_right = + 11/3 * u̅ᵢ    - 10/3 * uᵢ₊₁  + 2/3 * u̅ᵢ₊₁
p₁_right = +  2/3 * u̅ᵢ₋₁  +  2/3 * uᵢ    - 1/3 * u̅ᵢ    
p₂_right = -  1/3 * u̅ᵢ₋₂  +  2/3 * uᵢ₋₁  + 2/3 * u̅ᵢ₋₁  


p₂_left = 2/3  - 10/3 + 11/3
                                     
"""

struct WENO5S <: AbstractUpwindBiasedAdvectionScheme{2} end

@inline boundary_buffer(::WENO5S) = 2

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5S, c) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5S, c) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5S, c) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, centered_fourth_order, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::WENO5S, u) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, centered_fourth_order, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::WENO5S, v) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, centered_fourth_order, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::WENO5S, w) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, centered_fourth_order, w)

# Stencil to calculate the stretched WENO weights and smoothness indicators

@inline left_stencil_x(::Val{0}, i, j, k, ψ) = @inbounds ( (ψ[i-3, j, k], ψ[i-2, j, k], ψ[i-1, j, k]), (ψ[i-2, j, k], ψ[i-1, j, k], ψ[i, j, k]), (ψ[i-1, j, k], ψ[i, j, k], ψ[i+1, j, k]) )
@inline left_stencil_y(::Val{0}, i, j, k, ψ) = @inbounds ( (ψ[i, j-3, k], ψ[i, j-2, k], ψ[i, j-1, k]), (ψ[i, j-2, k], ψ[i, j-1, k], ψ[i, j, k]), (ψ[i, j-1, k], ψ[i, j, k], ψ[i, j+1, k]) )
@inline left_stencil_z(::Val{0}, i, j, k, ψ) = @inbounds ( (ψ[i, j, k-3], ψ[i, j, k-2], ψ[i, j, k-1]), (ψ[i, j, k-2], ψ[i, j, k-1], ψ[i, j, k]), (ψ[i, j, k-1], ψ[i, j, k], ψ[i, j, k+1]) )

@inline right_stencil_x(::Val{0}, i, j, k, ψ) = @inbounds ( (ψ[i-2, j, k], ψ[i-1, j, k], ψ[i, j, k]), (ψ[i-1, j, k], ψ[i, j, k], ψ[i+1, j, k]), (ψ[i, j, k], ψ[i+1, j, k], ψ[i+2, j, k]) )
@inline right_stencil_y(::Val{0}, i, j, k, ψ) = @inbounds ( (ψ[i, j-2, k], ψ[i, j-1, k], ψ[i, j, k]), (ψ[i, j-1, k], ψ[i, j, k], ψ[i, j+1, k]), (ψ[i, j, k], ψ[i, j+1, k], ψ[i, j+2, k]) )
@inline right_stencil_z(::Val{0}, i, j, k, ψ) = @inbounds ( (ψ[i, j, k-2], ψ[i, j, k-1], ψ[i, j, k]), (ψ[i, j, k-1], ψ[i, j, k], ψ[i, j, k+1]), (ψ[i, j, k], ψ[i, j, k+1], ψ[i, j, k+2]) )

#####
##### Coefficients for stretched (and uniform) WENO
#####

@inline coeff_left_p₀(FT, ::Val{0}) = (FT(1/3), FT(5/6), - FT(1/6))
@inline coeff_left_p₁(FT, ::Val{0}) = (- FT(1/6), FT(5/6), FT(1/3))
@inline coeff_left_p₂(FT, ::Val{0}) = (FT(1/3), - FT(7/6),  FT(11/6))

@inline coeff_left_p₀(FT, ::Val{1}) = (FT(2/3), FT(2/3), - FT(1/3))
@inline coeff_left_p₁(FT, ::Val{1}) = (- FT(1/3), FT(4/6), FT(2/3))                   
@inline coeff_left_p₂(FT, ::Val{1}) = (FT(2/3), - FT(10/3), FT(11/3))
                                            
@inline coeff_right_p₀(args...) = reverse(coeff_left_p₂(args...))
@inline coeff_right_p₁(args...) = reverse(coeff_left_p₁(args...))
@inline coeff_right_p₂(args...) = reverse(coeff_left_p₀(args...))

#####
##### biased pₖ for û calculation
#####

@inline left_biased_p₀(FT, GT, ψ) = @inbounds sum(coeff_left_p₀(FT, GT) .* ψ)
@inline left_biased_p₁(FT, GT, ψ) = @inbounds sum(coeff_left_p₁(FT, GT) .* ψ)
@inline left_biased_p₂(FT, GT, ψ) = @inbounds sum(coeff_left_p₂(FT, GT) .* ψ)

@inline right_biased_p₀(FT, GT, ψ) = @inbounds sum(coeff_right_p₀(FT, GT) .* ψ)
@inline right_biased_p₁(FT, GT, ψ) = @inbounds sum(coeff_right_p₁(FT, GT) .* ψ)
@inline right_biased_p₂(FT, GT, ψ) = @inbounds sum(coeff_right_p₂(FT, GT) .* ψ)

#####
##### Jiang & Shu (1996) WENO smoothness indicators. See also Equation 2.63 in Shu (1998).
#####

@inline coeff_β₀(FT, ::Val{0}) = (FT(13/12) , FT(1/4))
@inline coeff_β₁(FT, ::Val{0}) = (FT(13/12) , FT(1/4))
@inline coeff_β₂(FT, ::Val{0}) = (FT(13/12) , FT(1/4))

@inline coeff_β₀(FT, ::Val{1}) = (FT(26/12) , FT(1/2))
@inline coeff_β₁(FT, ::Val{1}) = (FT(26/12) , FT(1/2))
@inline coeff_β₂(FT, ::Val{1}) = (FT(26/12) , FT(1/2))

@inline left_biased_β₀(FT, GT, ψ) = @inbounds coeff_β₀(FT, GT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₀(FT, GT)[2] * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32
@inline left_biased_β₁(FT, GT, ψ) = @inbounds coeff_β₁(FT, GT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₁(FT, GT)[2] * ( ψ[1]         -  ψ[3])^two_32
@inline left_biased_β₂(FT, GT, ψ) = @inbounds coeff_β₂(FT, GT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₂(FT, GT)[2] * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32

@inline right_biased_β₀(FT, GT, ψ) = @inbounds coeff_β₀(FT, GT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₀(FT, GT)[2] * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32
@inline right_biased_β₁(FT, GT, ψ) = @inbounds coeff_β₁(FT, GT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₁(FT, GT)[2] * ( ψ[1]         -  ψ[3])^two_32
@inline right_biased_β₂(FT, GT, ψ) = @inbounds coeff_β₂(FT, GT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₂(FT, GT)[2] * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32

# Right-biased smoothness indicators are a reflection or "symmetric modification" of the left-biased smoothness
# indicators around grid point `i-1/2`.

@inline left_biased_α₀(FT, GT, ψ) = FT(C3₀) / (left_biased_β₀(FT, GT, ψ) + FT(ε))^ƞ
@inline left_biased_α₁(FT, GT, ψ) = FT(C3₁) / (left_biased_β₁(FT, GT, ψ) + FT(ε))^ƞ
@inline left_biased_α₂(FT, GT, ψ) = FT(C3₂) / (left_biased_β₂(FT, GT, ψ) + FT(ε))^ƞ

@inline right_biased_α₀(FT, GT, ψ) = FT(C3₀) / (right_biased_β₀(FT, GT, ψ) + FT(ε))^ƞ
@inline right_biased_α₁(FT, GT, ψ) = FT(C3₁) / (right_biased_β₁(FT, GT, ψ) + FT(ε))^ƞ
@inline right_biased_α₂(FT, GT, ψ) = FT(C3₂) / (right_biased_β₂(FT, GT, ψ) + FT(ε))^ƞ

#####
##### WENO-5 reconstruction
#####

@inline function left_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    α₀ = left_biased_α₀(FT, GT, ψ₀)
    α₁ = left_biased_α₁(FT, GT, ψ₁)
    α₂ = left_biased_α₂(FT, GT, ψ₂)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function right_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    α₀ = right_biased_α₀(FT, GT, ψ₀)
    α₁ = right_biased_α₁(FT, GT, ψ₁)
    α₂ = right_biased_α₂(FT, GT, ψ₂)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5S, ψ)
    
    typeof(grid) <: XRegRectilinearGrid ? GT = Val(1) : GT = Val(0)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = left_stencil_x(GT, i, j, k, ψ)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    return w₀ * left_biased_p₀(FT, GT, ψ₀) + 
           w₁ * left_biased_p₁(FT, GT, ψ₁) + 
           w₂ * left_biased_p₂(FT, GT, ψ₂)
end

@inline function left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5S, ψ)
    
    typeof(grid) <: YRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = left_stencil_y(GT, i, j, k, ψ)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    return w₀ * left_biased_p₀(FT, GT, ψ₀) + 
           w₁ * left_biased_p₁(FT, GT, ψ₁) + 
           w₂ * left_biased_p₂(FT, GT, ψ₂)
end

@inline function left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5S, ψ)
    
    typeof(grid) <: ZRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = left_stencil_z(GT, i, j, k, ψ)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    return w₀ * left_biased_p₀(FT, GT, ψ₀) +
           w₁ * left_biased_p₁(FT, GT, ψ₁) + 
           w₂ * left_biased_p₂(FT, GT, ψ₂)
end

@inline function right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5S, ψ)
    
    typeof(grid) <: XRegRectilinearGrid ? GT = Val(1) : GT = Val(0)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = right_stencil_x(GT, i, j, k, ψ)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    return w₀ * right_biased_p₀(FT, GT, ψ₀) +
           w₁ * right_biased_p₁(FT, GT, ψ₁) +
           w₂ * right_biased_p₂(FT, GT, ψ₂)
end

@inline function right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5S, ψ)
    
    typeof(grid) <: YRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = right_stencil_y(GT, i, j, k, ψ)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    return w₀ * right_biased_p₀(FT, GT, ψ₀) + 
           w₁ * right_biased_p₁(FT, GT, ψ₁) + 
           w₂ * right_biased_p₂(FT, GT, ψ₂)
end

@inline function right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5S, ψ)
    
    typeof(grid) <: ZRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = right_stencil_z(GT, i, j, k, ψ)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    return w₀ * right_biased_p₀(FT, GT, ψ₀) + 
           w₁ * right_biased_p₁(FT, GT, ψ₁) +
           w₂ * right_biased_p₂(FT, GT, ψ₂)
end

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5S, ψ) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5S, ψ) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5S, ψ) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5S, ψ) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5S, ψ) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5S, ψ) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ)


function right_stencil_x(::Val{1}, i, j, k, ψ, grid::AG{FT}) where FT
    u̅ = zeros(FT, 4)

    for h = -2:1
        γ =  Δxᶜᶜᵃ(i+h, j, k, grid)/(Δxᶜᶜᵃ(i+h, j, k, grid) + Δxᶜᶜᵃ(i+h+1, j, k, grid))
        u̅[h+3] = (1 - γ) * ψ[i+h, j, k] + γ * ψ[i+h+1, j, k]
    end

    return ((u̅[1], ψ[i-1, j, k], u̅[2]),
            (u̅[2], ψ[i, j, k]  , u̅[3]),
            (u̅[3], ψ[i+1, j, k], u̅[4]))
end

function left_stencil_x(::Val{1}, i, j, k, ψ, grid::AG{FT}) where FT
    u̅ = zeros(FT, 4)

    for h = -3:0
        γ =  Δxᶜᶜᵃ(i+h, j, k, grid)/(Δxᶜᶜᵃ(i+h, j, k, grid) + Δxᶜᶜᵃ(i+h+1, j, k, grid))
        u̅[h+4] = (1 - γ) * ψ[i+h, j, k] + γ * ψ[i+h+1, j, k]
    end

    u⁺ = ((2 * Δxᶜᶜᵃ(i-2, j, k, grid) + Δxᶜᶜᵃ(i-3, j, k, grid)) * ψ[i-2, j, k] - Δxᶜᶜᵃ(i-2, j, k, grid) * ψ[i-3, j, k]) / 
              (Δxᶜᶜᵃ(i-2, j, k, grid) + Δxᶜᶜᵃ(i-3, j, k, grid))

    return ((u̅[1], ψ[i-2, j, k], u̅[2]), #(u⁺,   u̅[1], ψ[i-2, j, k]), #
            (u̅[2], ψ[i-1, j, k], u̅[3]),
            (u̅[3], ψ[i, j, k]  , u̅[4]))
end
