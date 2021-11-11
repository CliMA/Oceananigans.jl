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

@inline left_stencil_x(::Val{0}, i, j, k, ψ, args...) = @inbounds ( (ψ[i-3, j, k], ψ[i-2, j, k], ψ[i-1, j, k]), (ψ[i-2, j, k], ψ[i-1, j, k], ψ[i, j, k]), (ψ[i-1, j, k], ψ[i, j, k], ψ[i+1, j, k]) )
@inline left_stencil_y(::Val{0}, i, j, k, ψ, args...) = @inbounds ( (ψ[i, j-3, k], ψ[i, j-2, k], ψ[i, j-1, k]), (ψ[i, j-2, k], ψ[i, j-1, k], ψ[i, j, k]), (ψ[i, j-1, k], ψ[i, j, k], ψ[i, j+1, k]) )
@inline left_stencil_z(::Val{0}, i, j, k, ψ, args...) = @inbounds ( (ψ[i, j, k-3], ψ[i, j, k-2], ψ[i, j, k-1]), (ψ[i, j, k-2], ψ[i, j, k-1], ψ[i, j, k]), (ψ[i, j, k-1], ψ[i, j, k], ψ[i, j, k+1]) )

@inline right_stencil_x(::Val{0}, i, j, k, ψ, args...) = @inbounds ( (ψ[i-2, j, k], ψ[i-1, j, k], ψ[i, j, k]), (ψ[i-1, j, k], ψ[i, j, k], ψ[i+1, j, k]), (ψ[i, j, k], ψ[i+1, j, k], ψ[i+2, j, k]) )
@inline right_stencil_y(::Val{0}, i, j, k, ψ, args...) = @inbounds ( (ψ[i, j-2, k], ψ[i, j-1, k], ψ[i, j, k]), (ψ[i, j-1, k], ψ[i, j, k], ψ[i, j+1, k]), (ψ[i, j, k], ψ[i, j+1, k], ψ[i, j+2, k]) )
@inline right_stencil_z(::Val{0}, i, j, k, ψ, args...) = @inbounds ( (ψ[i, j, k-2], ψ[i, j, k-1], ψ[i, j, k]), (ψ[i, j, k-1], ψ[i, j, k], ψ[i, j, k+1]), (ψ[i, j, k], ψ[i, j, k+1], ψ[i, j, k+2]) )

#####
##### Coefficients for stretched (and uniform) WENO
#####

@inline coeff_left_p₀(FT, ::Val{0}) = (FT(1/3), FT(5/6), - FT(1/6))
@inline coeff_left_p₁(FT, ::Val{0}) = (- FT(1/6), FT(5/6), FT(1/3))
@inline coeff_left_p₂(FT, ::Val{0}) = (FT(1/3), - FT(7/6),  FT(11/6))
                                          
@inline coeff_left_p₀(FT, ::Val{1}) = (FT(1/3), FT(5/6), - FT(1/6))
@inline coeff_left_p₁(FT, ::Val{1}) = (- FT(1/6), FT(5/6), FT(1/3))
@inline coeff_left_p₂(FT, ::Val{1}) = (FT(1/3), - FT(7/6),  FT(11/6))

@inline coeff_right_p₀(args...) = reverse(coeff_left_p₂(args...))
@inline coeff_right_p₁(args...) = reverse(coeff_left_p₁(args...))
@inline coeff_right_p₂(args...) = reverse(coeff_left_p₀(args...))

#####
##### biased pₖ for û calculation
#####

@inline left_biased_p₀(FT, GT, ψ, args...) = @inbounds sum(coeff_left_p₀(FT, GT) .* ψ)
@inline left_biased_p₁(FT, GT, ψ, args...) = @inbounds sum(coeff_left_p₁(FT, GT) .* ψ)
@inline left_biased_p₂(FT, GT, ψ, args...) = @inbounds sum(coeff_left_p₂(FT, GT) .* ψ)

@inline right_biased_p₀(FT, GT, ψ, args...) = @inbounds sum(coeff_right_p₀(FT, GT) .* ψ)
@inline right_biased_p₁(FT, GT, ψ, args...) = @inbounds sum(coeff_right_p₁(FT, GT) .* ψ)
@inline right_biased_p₂(FT, GT, ψ, args...) = @inbounds sum(coeff_right_p₂(FT, GT) .* ψ)

@inline left_biased_p₀(FT, ::Val{1}, ψ, cf, c, i) = @inbounds sum(left_L₀(cf, c, i) .* ψ) + left_C₀(FT, ψ, c, i)
@inline left_biased_p₁(FT, ::Val{1}, ψ, cf, c, i) = @inbounds sum(left_L₁(cf, c, i) .* ψ) + left_C₁(FT, ψ, c, i)
@inline left_biased_p₂(FT, ::Val{1}, ψ, cf, c, i) = @inbounds sum(left_L₂(cf, c, i) .* ψ) + left_C₂(FT, ψ, c, i)

##₁###
##₂### Jiang & Shu (1996) WENO smoothness indicators. See also Equation 2.63 in Shu (1998)
#####

@inline coeff_β₀(FT) = (FT(13/12) , FT(0.25))
@inline coeff_β₁(FT) = (FT(13/12) , FT(0.25))
@inline coeff_β₂(FT) = (FT(13/12) , FT(0.25))

@inline left_biased_β₀(FT, ψ) = @inbounds coeff_β₀(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₀(FT)[2] * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32
@inline left_biased_β₁(FT, ψ) = @inbounds coeff_β₁(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₁(FT)[2] * ( ψ[1]         -  ψ[3])^two_32
@inline left_biased_β₂(FT, ψ) = @inbounds coeff_β₂(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₂(FT)[2] * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32

@inline right_biased_β₀(FT, ψ) = @inbounds coeff_β₀(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₀(FT)[2] * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32
@inline right_biased_β₁(FT, ψ) = @inbounds coeff_β₁(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₁(FT)[2] * ( ψ[1]         -  ψ[3])^two_32
@inline right_biased_β₂(FT, ψ) = @inbounds coeff_β₂(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₂(FT)[2] * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32

# Right-biased smoothness indicators are a reflection or "symmetric modification" of the left-biased smoothness
# indicators around grid point `i-1/2`.

@inline left_biased_α₀(FT, ψ) = FT(C3₀) / (left_biased_β₀(FT, ψ) + FT(ε))^ƞ
@inline left_biased_α₁(FT, ψ) = FT(C3₁) / (left_biased_β₁(FT, ψ) + FT(ε))^ƞ
@inline left_biased_α₂(FT, ψ) = FT(C3₂) / (left_biased_β₂(FT, ψ) + FT(ε))^ƞ

@inline right_biased_α₀(FT, ψ) = FT(C3₀) / (right_biased_β₀(FT, ψ) + FT(ε))^ƞ
@inline right_biased_α₁(FT, ψ) = FT(C3₁) / (right_biased_β₁(FT, ψ) + FT(ε))^ƞ
@inline right_biased_α₂(FT, ψ) = FT(C3₂) / (right_biased_β₂(FT, ψ) + FT(ε))^ƞ

#####
##### WENO-5 reconstruction
#####

@inline function left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    α₀ = left_biased_α₀(FT, ψ₀)
    α₁ = left_biased_α₁(FT, ψ₁)
    α₂ = left_biased_α₂(FT, ψ₂)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    α₀ = right_biased_α₀(FT, ψ₀)
    α₁ = right_biased_α₁(FT, ψ₁)
    α₂ = right_biased_α₂(FT, ψ₂)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5S, ψ)
    
    typeof(grid) <: XRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    GT = Val(0)
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = left_stencil_x(i, j, k, ψ, grid)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    return w₀ * left_biased_p₀(FT, GT, ψ₀, grid.xᶠᵃᵃ[i], grid.xᶜᵃᵃ, i) + 
           w₁ * left_biased_p₁(FT, GT, ψ₁, grid.xᶠᵃᵃ[i], grid.xᶜᵃᵃ, i) + 
           w₂ * left_biased_p₂(FT, GT, ψ₂, grid.xᶠᵃᵃ[i], grid.xᶜᵃᵃ, i)
end

@inline function left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5S, ψ)
    
    typeof(grid) <: YRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = left_stencil_y(i, j, k, ψ, grid)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    return w₀ * left_biased_p₀(FT, GT, ψ₀, grid.yᵃᶠᵃ[j], grid.yᵃᶜᵃ, j) + 
           w₁ * left_biased_p₁(FT, GT, ψ₁, grid.yᵃᶠᵃ[j], grid.yᵃᶜᵃ, j) + 
           w₂ * left_biased_p₂(FT, GT, ψ₂, grid.yᵃᶠᵃ[j], grid.yᵃᶜᵃ, j)
end

@inline function left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5S, ψ)
    
    typeof(grid) <: ZRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = left_stencil_z(i, j, k, ψ, grid)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    return w₀ * left_biased_p₀(FT, GT, ψ₀, grid.zᵃᵃᶠ[k], grid.zᵃᵃᶜ, k) +
           w₁ * left_biased_p₁(FT, GT, ψ₁, grid.zᵃᵃᶠ[k], grid.zᵃᵃᶜ, k) + 
           w₂ * left_biased_p₂(FT, GT, ψ₂, grid.zᵃᵃᶠ[k], grid.zᵃᵃᶜ, k)
end

@inline function right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5S, ψ)
    
    typeof(grid) <: XRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    GT = Val(0)
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = right_stencil_x(i, j, k, ψ, grid)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    return w₀ * right_biased_p₀(FT, GT, ψ₀, grid.xᶠᵃᵃ[i], grid.xᶜᵃᵃ, i) +
           w₁ * right_biased_p₁(FT, GT, ψ₁, grid.xᶠᵃᵃ[i], grid.xᶜᵃᵃ, i) +
           w₂ * right_biased_p₂(FT, GT, ψ₂, grid.xᶠᵃᵃ[i], grid.xᶜᵃᵃ, i)
end

@inline function right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5S, ψ)
    
    typeof(grid) <: YRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = right_stencil_y(i, j, k, ψ, grid)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    return w₀ * right_biased_p₀(FT, GT, ψ₀, grid.yᵃᶠᵃ[j], grid.yᵃᶜᵃ, j) + 
           w₁ * right_biased_p₁(FT, GT, ψ₁, grid.yᵃᶠᵃ[j], grid.yᵃᶜᵃ, j) + 
           w₂ * right_biased_p₂(FT, GT, ψ₂, grid.yᵃᶠᵃ[j], grid.yᵃᶜᵃ, j)
end

@inline function right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5S, ψ)
    
    typeof(grid) <: ZRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = right_stencil_z(i, j, k, ψ, grid)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    return w₀ * right_biased_p₀(FT, GT, ψ₀, grid.zᵃᵃᶠ[k], grid.zᵃᵃᶜ, k) + 
           w₁ * right_biased_p₁(FT, GT, ψ₁, grid.zᵃᵃᶠ[k], grid.zᵃᵃᶜ, k) +
           w₂ * right_biased_p₂(FT, GT, ψ₂, grid.zᵃᵃᶠ[k], grid.zᵃᵃᶜ, k)
end

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5S, ψ) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5S, ψ) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5S, ψ) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5S, ψ) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5S, ψ) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5S, ψ) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ)


@inline left_L₀(cf, c, i) = ( (cf - c[i]  ) * (cf - c[i+1]) / (c[i-1] - c[i]  ) / (c[i-1] - c[i+1]), 
                              (cf - c[i-1]) * (cf - c[i+1]) / (c[i-1] - c[i]  ) / (c[i-1] - c[i+1]),
                              (cf - c[i-1]) * (cf - c[i]  ) / (c[i+1] - c[i-1]) / (c[i+1] - c[i]  ) )
@inline left_L₁(cf, c, i) = ( (cf - c[i-1]) * (cf - c[i]  ) / (c[i-2] - c[i-1]) / (c[i-2] - c[i]), 
                              (cf - c[i-2]) * (cf - c[i]  ) / (c[i-2] - c[i-1]) / (c[i-2] - c[i]),
                              (cf - c[i-2]) * (cf - c[i-1]) / (c[i]   - c[i-2]) / (c[i]   - c[i-1]) )
@inline left_L₂(cf, c, i) = ( (cf - c[i-2]) * (cf - c[i-1]) / (c[i-3] - c[i-2]) / (c[i-3] - c[i-1]), 
                              (cf - c[i-3]) * (cf - c[i-1]) / (c[i-3] - c[i-2]) / (c[i-3] - c[i-1]),
                              (cf - c[i-3]) * (cf - c[i-2]) / (c[i-1] - c[i-3]) / (c[i-1] - c[i-2]) )

@inline left_C₀(FT, ψ, c, i) = - ((c[i]   - c[i-1]) * ψ[3] - (c[i+1] - c[i-1]) * ψ[2] + (c[i+1] - c[i]  ) * ψ[1]) / (c[i+1] - c[i]  ) / FT(12)
@inline left_C₁(FT, ψ, c, i) = - ((c[i-1] - c[i-2]) * ψ[3] - (c[i]   - c[i-2]) * ψ[2] + (c[i]   - c[i-1]) * ψ[1]) / (c[i]   - c[i-1]) / FT(12)
@inline left_C₂(FT, ψ, c, i) = - ((c[i-2] - c[i-3]) * ψ[3] - (c[i-1] - c[i-3]) * ψ[2] + (c[i-1] - c[i-2]) * ψ[1]) / (c[i-1] - c[i-2]) / FT(12)

