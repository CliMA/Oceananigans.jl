#####
##### Weighted Essentially Non-Oscillatory (WENO) scheme of order 5
#####

"""

GT here is used as a variable to decide wether to dispatch 
on a "regular formulation" or a "stretched formulation"

GT can take up the value of 
- Val(0) => regular weno formulation
- Val(1) => stretched weno formulation
                                     
"""

struct WENO5S <: AbstractUpwindBiasedAdvectionScheme{2} end

@inline boundary_buffer(::WENO5S) = 2

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5S, c) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5S, c) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5S, c) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, centered_fourth_order, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::WENO5S, u) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, centered_fourth_order, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::WENO5S, v) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, centered_fourth_order, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::WENO5S, w) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, centered_fourth_order, w)

# Unroll the functions to pass the coordinates in case of a stretched grid
@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO5S, ψ)  = weno_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, grid.xᶠᵃᵃ, i)
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO5S, ψ)  = weno_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, grid.yᵃᶠᵃ, j)
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO5S, ψ)  = weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, grid.zᵃᵃᶠ, k)

@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO5S, ψ) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, grid.yᵃᶠᵃ, i)
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO5S, ψ) = weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, grid.zᵃᵃᶠ, j)
@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO5S, ψ) = weno_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, grid.xᶠᵃᵃ, k)

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5S, ψ)  = weno_left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, grid.xᶜᵃᵃ, i)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5S, ψ)  = weno_left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, grid.yᵃᶜᵃ, j)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5S, ψ)  = weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, grid.zᵃᵃᶜ, k)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5S, ψ) = weno_right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, grid.xᶜᵃᵃ, i)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5S, ψ) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, grid.yᵃᶜᵃ, j)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5S, ψ) = weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, grid.zᵃᵃᶜ, k)

# Stencil to calculate the stretched WENO weights and smoothness indicators
@inline left_stencil_x(i, j, k, ψ, args...) = @inbounds ( (ψ[i-3, j, k], ψ[i-2, j, k], ψ[i-1, j, k]), (ψ[i-2, j, k], ψ[i-1, j, k], ψ[i, j, k]), (ψ[i-1, j, k], ψ[i, j, k], ψ[i+1, j, k]) )
@inline left_stencil_y(i, j, k, ψ, args...) = @inbounds ( (ψ[i, j-3, k], ψ[i, j-2, k], ψ[i, j-1, k]), (ψ[i, j-2, k], ψ[i, j-1, k], ψ[i, j, k]), (ψ[i, j-1, k], ψ[i, j, k], ψ[i, j+1, k]) )
@inline left_stencil_z(i, j, k, ψ, args...) = @inbounds ( (ψ[i, j, k-3], ψ[i, j, k-2], ψ[i, j, k-1]), (ψ[i, j, k-2], ψ[i, j, k-1], ψ[i, j, k]), (ψ[i, j, k-1], ψ[i, j, k], ψ[i, j, k+1]) )

@inline right_stencil_x(i, j, k, ψ, args...) = @inbounds ( (ψ[i-2, j, k], ψ[i-1, j, k], ψ[i, j, k]), (ψ[i-1, j, k], ψ[i, j, k], ψ[i+1, j, k]), (ψ[i, j, k], ψ[i+1, j, k], ψ[i+2, j, k]) )
@inline right_stencil_y(i, j, k, ψ, args...) = @inbounds ( (ψ[i, j-2, k], ψ[i, j-1, k], ψ[i, j, k]), (ψ[i, j-1, k], ψ[i, j, k], ψ[i, j+1, k]), (ψ[i, j, k], ψ[i, j+1, k], ψ[i, j+2, k]) )
@inline right_stencil_z(i, j, k, ψ, args...) = @inbounds ( (ψ[i, j, k-2], ψ[i, j, k-1], ψ[i, j, k]), (ψ[i, j, k-1], ψ[i, j, k], ψ[i, j, k+1]), (ψ[i, j, k], ψ[i, j, k+1], ψ[i, j, k+2]) )

#####
##### biased pₖ for û calculation
#####

@inline left_biased_p₀(FT, GT, ψ, args...) = @inbounds sum(coeff_left_p₀(FT, GT, args...) .* ψ)
@inline left_biased_p₁(FT, GT, ψ, args...) = @inbounds sum(coeff_left_p₁(FT, GT, args...) .* ψ)
@inline left_biased_p₂(FT, GT, ψ, args...) = @inbounds sum(coeff_left_p₂(FT, GT, args...) .* ψ)

@inline right_biased_p₀(FT, GT, ψ, args...) = @inbounds sum(coeff_right_p₀(FT, GT, args...) .* ψ)
@inline right_biased_p₁(FT, GT, ψ, args...) = @inbounds sum(coeff_right_p₁(FT, GT, args...) .* ψ)
@inline right_biased_p₂(FT, GT, ψ, args...) = @inbounds sum(coeff_right_p₂(FT, GT, args...) .* ψ)

##₁###
##₂### Jiang & Shu (1996) WENO smoothness indicators. See also Equation 2.63 in Shu (1998)
#####

@inline coeff_β₀(FT) = (FT(13/12) , FT(0.25))
@inline coeff_β₁(FT) = (FT(13/12) , FT(0.25))
@inline coeff_β₂(FT) = (FT(13/12) , FT(0.25))

@inline left_biased_β₀(FT, GT, ψ, args...) = @inbounds coeff_β₀(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₀(FT)[2] * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32
@inline left_biased_β₁(FT, GT, ψ, args...) = @inbounds coeff_β₁(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₁(FT)[2] * ( ψ[1]         -  ψ[3])^two_32
@inline left_biased_β₂(FT, GT, ψ, args...) = @inbounds coeff_β₂(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₂(FT)[2] * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32

@inline right_biased_β₀(FT, GT, ψ, args...) = @inbounds coeff_β₀(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₀(FT)[2] * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32
@inline right_biased_β₁(FT, GT, ψ, args...) = @inbounds coeff_β₁(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₁(FT)[2] * ( ψ[1]         -  ψ[3])^two_32
@inline right_biased_β₂(FT, GT, ψ, args...) = @inbounds coeff_β₂(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₂(FT)[2] * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32

# Right-biased smoothness indicators are a reflection or "symmetric modification" of the left-biased smoothness
# indicators around grid point `i-1/2`.

@inline left_biased_α₀(FT, GT, ψ, args...) = FT(C3₀) / (left_biased_β₀(FT, GT, ψ, args...) + FT(ε))^ƞ
@inline left_biased_α₁(FT, GT, ψ, args...) = FT(C3₁) / (left_biased_β₁(FT, GT, ψ, args...) + FT(ε))^ƞ
@inline left_biased_α₂(FT, GT, ψ, args...) = FT(C3₂) / (left_biased_β₂(FT, GT, ψ, args...) + FT(ε))^ƞ

@inline right_biased_α₀(FT, GT, ψ, args...) = FT(C3₀) / (right_biased_β₀(FT, GT, ψ, args...) + FT(ε))^ƞ
@inline right_biased_α₁(FT, GT, ψ, args...) = FT(C3₁) / (right_biased_β₁(FT, GT, ψ, args...) + FT(ε))^ƞ
@inline right_biased_α₂(FT, GT, ψ, args...) = FT(C3₂) / (right_biased_β₂(FT, GT, ψ, args...) + FT(ε))^ƞ

#####
##### WENO-5 reconstruction
#####

@inline function left_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀, args...)
    α₀ = left_biased_α₀(FT, GT, ψ₀, args...)
    α₁ = left_biased_α₁(FT, GT, ψ₁, args...)
    α₂ = left_biased_α₂(FT, GT, ψ₂, args...)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function right_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀, args...)
    α₁ = right_biased_α₁(FT, GT, ψ₁, args...)
    α₂ = right_biased_α₂(FT, GT, ψ₂, args...)
    α₀ = right_biased_α₀(FT, GT, ψ₀, args...)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

#####
##### Coefficients for stretched (and uniform) WENO
#####
                                         
@inline coeff_left_p₀(FT, ::Val{0}, args... ) = (FT(1/3), FT(5/6), - FT(1/6))   
@inline coeff_left_p₁(FT, ::Val{0}, args... ) = (- FT(1/6), FT(5/6), FT(1/3))   
@inline coeff_left_p₂(FT, ::Val{0}, args... ) = (FT(1/3), - FT(7/6),  FT(11/6)) 

@inline coeff_left_p₀(FT, ::Val{1}, args... ) = interpolation_weights(0, args...)   
@inline coeff_left_p₁(FT, ::Val{1}, args... ) = interpolation_weights(1, args...)   
@inline coeff_left_p₂(FT, ::Val{1}, args... ) = interpolation_weights(2, args...)   

@inline coeff_right_p₀(FT, ::Val{0}, args...) = reverse(coeff_left_p₂(FT, Val(0), args...)) 
@inline coeff_right_p₁(FT, ::Val{0}, args...) = reverse(coeff_left_p₁(FT, Val(0), args...)) 
@inline coeff_right_p₂(FT, ::Val{0}, args...) = reverse(coeff_left_p₀(FT, Val(0), args...)) 

@inline coeff_right_p₀(FT, ::Val{1}, args...) = interpolation_weights(-1, args...) 
@inline coeff_right_p₁(FT, ::Val{1}, args...) = interpolation_weights( 0, args...) 
@inline coeff_right_p₂(FT, ::Val{1}, args...) = interpolation_weights( 1, args...) 


@inline function weno_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5S, ψ, args...)
    
    typeof(grid) <: XRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = left_stencil_x(i, j, k, ψ, grid)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    return w₀ * left_biased_p₀(FT, GT, ψ₀, args...) + 
           w₁ * left_biased_p₁(FT, GT, ψ₁, args...) + 
           w₂ * left_biased_p₂(FT, GT, ψ₂, args...)
end

@inline function weno_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5S, ψ, args...)
    
    typeof(grid) <: YRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = left_stencil_y(i, j, k, ψ, grid)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    return w₀ * left_biased_p₀(FT, GT, ψ₀, args...) + 
           w₁ * left_biased_p₁(FT, GT, ψ₁, args...) + 
           w₂ * left_biased_p₂(FT, GT, ψ₂, args...)
end

@inline function weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5S, ψ, args...)
    
    typeof(grid) <: ZRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = left_stencil_z(i, j, k, ψ, grid)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    return w₀ * left_biased_p₀(FT, GT, ψ₀, args...) +
           w₁ * left_biased_p₁(FT, GT, ψ₁, args...) + 
           w₂ * left_biased_p₂(FT, GT, ψ₂, args...)
end

@inline function weno_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5S, ψ, args...)
    
    typeof(grid) <: XRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = right_stencil_x(i, j, k, ψ, grid)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    return w₀ * right_biased_p₀(FT, GT, ψ₀, args...) +
           w₁ * right_biased_p₁(FT, GT, ψ₁, args...) +
           w₂ * right_biased_p₂(FT, GT, ψ₂, args...)
end

@inline function weno_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5S, ψ, args...)
    
    typeof(grid) <: YRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = right_stencil_y(i, j, k, ψ, grid)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    return w₀ * right_biased_p₀(FT, GT, ψ₀, args...) + 
           w₁ * right_biased_p₁(FT, GT, ψ₁, args...) + 
           w₂ * right_biased_p₂(FT, GT, ψ₂, args...)
end

@inline function weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5S, ψ, args...)
    
    typeof(grid) <: ZRegRectilinearGrid ? GT = Val(0) : GT = Val(1)
    
    FT         = eltype(grid)
    ψ₂, ψ₁, ψ₀ = right_stencil_z(i, j, k, ψ, grid)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, GT, ψ₂, ψ₁, ψ₀)
    return w₀ * right_biased_p₀(FT, GT, ψ₀, args...) + 
           w₁ * right_biased_p₁(FT, GT, ψ₁, args...) +
           w₂ * right_biased_p₂(FT, GT, ψ₂, args...)
end

@inline function interpolation_weights(r, coord, i)

    coeff = ()
    @inbounds begin
    for j=0:2
        c = 0
        for m = j+1:3
            num = 0
            for l = 0:3
                if l != m
                    prod = 1
                    for q = 0:3
                        if q != m && q != l 
                            prod *= (coord[i] - coord[i-r+q-1])
                        end
                    end
                    num += prod
                end
            end
            den = 1
            for l = 0:3
                if l!= m
                    den *= (coord[i-r+m-1] - coord[i-r+l-1])
                end
            end
            c += num / den
        end 
        coeff = (coeff..., c * (coord[i-r+j] - coord[i-r+j-1]))
    end
    end

    return coeff
end