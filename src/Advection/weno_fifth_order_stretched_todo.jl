#####
##### Weighted Essentially Non-Oscillatory (WENO) scheme of order 5
#####

using OffsetArrays
using Oceananigans.Architectures: arch_array
import Base: show

struct WENO5S{FT, XT, YT, ZT, Buffer} <: AbstractUpwindBiasedAdvectionScheme{2} 
    
    coeff_xᶠᵃᵃ::XT
    coeff_xᶜᵃᵃ::XT
    
    coeff_yᵃᶠᵃ::YT
    coeff_yᵃᶜᵃ::YT
    
    coeff_zᵃᵃᶠ::ZT
    coeff_zᵃᵃᶜ::ZT
 
end

function WENO5S(FT = Float64; grid = nothing) 
    
    if grid isa Nothing
        coeff_xᶠᵃᵃ = nothing 
        coeff_xᶜᵃᵃ = nothing  
        coeff_yᵃᶠᵃ = nothing   
        coeff_yᵃᶜᵃ = nothing
        coeff_zᵃᵃᶠ = nothing
        coeff_zᵃᵃᶜ = nothing
    elseif !(typeof(grid) <: RectilinearGrid)
        @warn "Stretched WENO is not supported with grids other than Rectilinear, defaulting to Uniform WENO"
        coeff_xᶠᵃᵃ = nothing 
        coeff_xᶜᵃᵃ = nothing  
        coeff_yᵃᶠᵃ = nothing   
        coeff_yᵃᶜᵃ = nothing
        coeff_zᵃᵃᶠ = nothing
        coeff_zᵃᵃᶜ = nothing    
    else
        FT = eltype(grid)
        
        arch = grid.architecture
        
        if typeof(grid) <: XRegRectilinearGrid 
            coeff_xᶠᵃᵃ = nothing 
            coeff_xᶜᵃᵃ = nothing 
        else
            coeff_xᶠᵃᵃ = calc_interpolating_coefficients(FT, grid.xᶠᵃᵃ, arch, grid.Nx) 
            coeff_xᶜᵃᵃ = calc_interpolating_coefficients(FT, grid.xᶜᵃᵃ, arch, grid.Nx)
        end
        if typeof(grid) <: YRegRectilinearGrid 
            coeff_yᵃᶠᵃ = nothing   
            coeff_yᵃᶜᵃ = nothing
        else    
            coeff_yᵃᶠᵃ = calc_interpolating_coefficients(FT, grid.yᵃᶠᵃ, arch, grid.Ny)
            coeff_yᵃᶜᵃ = calc_interpolating_coefficients(FT, grid.yᵃᶜᵃ, arch, grid.Ny)
        end
        if typeof(grid) <: ZRegRectilinearGrid 
            coeff_zᵃᵃᶠ = nothing
            coeff_zᵃᵃᶜ = nothing
        else
            coeff_zᵃᵃᶠ = calc_interpolating_coefficients(FT, grid.zᵃᵃᶠ, arch, grid.Nz)
            coeff_zᵃᵃᶜ = calc_interpolating_coefficients(FT, grid.zᵃᵃᶜ, arch, grid.Nz)
        end
    end
    XT = typeof(coeff_xᶠᵃᵃ)
    YT = typeof(coeff_yᵃᶠᵃ)
    ZT = typeof(coeff_zᵃᵃᶠ)

    return WENO5S{FT, XT, YT, ZT, 2}(coeff_xᶠᵃᵃ, coeff_xᶜᵃᵃ, coeff_yᵃᶠᵃ, coeff_yᵃᶜᵃ, coeff_zᵃᵃᶠ, coeff_zᵃᵃᶜ)
end

function Base.show(io::IO, a::WENO5S{ FT, RX, RY, RZ, Buffer}) where {FT, RX, RY, RZ, Buffer}
    print(io, "WENO5 advection sheme with X $(RX == Nothing ? "regular" : "stretched"), Y $(RY == Nothing ? "regular" : "stretched") and Z $(RZ == Nothing ? "regular" : "stretched")")
end

@inline boundary_buffer(::WENO5S) = 2

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5S, c) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5S, c) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5S, c) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, centered_fourth_order, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::WENO5S, u) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, centered_fourth_order, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::WENO5S, v) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, centered_fourth_order, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::WENO5S, w) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, centered_fourth_order, w)

# Unroll the functions to pass the coordinates in case of a stretched grid
@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO5S, ψ)  = weno_left_biased_interpolate_xᶠᵃᵃ(i, j, k, scheme, ψ, i, Face)
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO5S, ψ)  = weno_left_biased_interpolate_yᵃᶠᵃ(i, j, k, scheme, ψ, j, Face)
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO5S, ψ)  = weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k, scheme, ψ, k, Face)

@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO5S, ψ) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j, k, scheme, ψ, i, Face)
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO5S, ψ) = weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k, scheme, ψ, j, Face)
@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO5S, ψ) = weno_right_biased_interpolate_xᶠᵃᵃ(i, j, k, scheme, ψ, k, Face)

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5S, ψ)  = weno_left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, scheme, ψ, i, Center)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5S, ψ)  = weno_left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, scheme, ψ, j, Center)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5S, ψ)  = weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, scheme, ψ, k, Center)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5S, ψ) = weno_right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, scheme, ψ, i, Center)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5S, ψ) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, scheme, ψ, j, Center)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5S, ψ) = weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, scheme, ψ, k, Center)

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

@inline left_biased_p₀(scheme, ψ, args...) = @inbounds sum(coeff_left_p₀(scheme, args...) .* ψ)
@inline left_biased_p₁(scheme, ψ, args...) = @inbounds sum(coeff_left_p₁(scheme, args...) .* ψ)
@inline left_biased_p₂(scheme, ψ, args...) = @inbounds sum(coeff_left_p₂(scheme, args...) .* ψ)

@inline right_biased_p₀(scheme, ψ, args...) = @inbounds sum(coeff_right_p₀(scheme, args...) .* ψ)
@inline right_biased_p₁(scheme, ψ, args...) = @inbounds sum(coeff_right_p₁(scheme, args...) .* ψ)
@inline right_biased_p₂(scheme, ψ, args...) = @inbounds sum(coeff_right_p₂(scheme, args...) .* ψ)

##₁###
##₂### Jiang & Shu (1996) WENO smoothness indicators. See also Equation 2.63 in Shu (1998)
#####

@inline coeff_β₀(FT) = (FT(13/12) , FT(0.25))
@inline coeff_β₁(FT) = (FT(13/12) , FT(0.25))
@inline coeff_β₂(FT) = (FT(13/12) , FT(0.25))

@inline left_biased_β₀(FT, ψ, args...) = @inbounds coeff_β₀(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₀(FT)[2] * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32
@inline left_biased_β₁(FT, ψ, args...) = @inbounds coeff_β₁(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₁(FT)[2] * ( ψ[1]         -  ψ[3])^two_32
@inline left_biased_β₂(FT, ψ, args...) = @inbounds coeff_β₂(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₂(FT)[2] * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32

@inline right_biased_β₀(FT, ψ, args...) = @inbounds coeff_β₀(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₀(FT)[2] * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32
@inline right_biased_β₁(FT, ψ, args...) = @inbounds coeff_β₁(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₁(FT)[2] * ( ψ[1]         -  ψ[3])^two_32
@inline right_biased_β₂(FT, ψ, args...) = @inbounds coeff_β₂(FT)[1] * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + coeff_β₂(FT)[2] * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32

# Right-biased smoothness indicators are a reflection or "symmetric modification" of the left-biased smoothness
# indicators around grid point `i-1/2`.

@inline left_biased_α₀(FT, ψ, args...) = FT(C3₀) / (left_biased_β₀(FT, ψ, args...) + FT(ε))^ƞ
@inline left_biased_α₁(FT, ψ, args...) = FT(C3₁) / (left_biased_β₁(FT, ψ, args...) + FT(ε))^ƞ
@inline left_biased_α₂(FT, ψ, args...) = FT(C3₂) / (left_biased_β₂(FT, ψ, args...) + FT(ε))^ƞ

@inline right_biased_α₀(FT, ψ, args...) = FT(C3₀) / (right_biased_β₀(FT, ψ, args...) + FT(ε))^ƞ
@inline right_biased_α₁(FT, ψ, args...) = FT(C3₁) / (right_biased_β₁(FT, ψ, args...) + FT(ε))^ƞ
@inline right_biased_α₂(FT, ψ, args...) = FT(C3₂) / (right_biased_β₂(FT, ψ, args...) + FT(ε))^ƞ

#####
##### WENO-5 reconstruction
#####

@inline function left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀, args...)
    α₀ = left_biased_α₀(FT, ψ₀, args...)
    α₁ = left_biased_α₁(FT, ψ₁, args...)
    α₂ = left_biased_α₂(FT, ψ₂, args...)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀, args...)
    α₁ = right_biased_α₁(FT, ψ₁, args...)
    α₂ = right_biased_α₂(FT, ψ₂, args...)
    α₀ = right_biased_α₀(FT, ψ₀, args...)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function weno_left_biased_interpolate_xᶠᵃᵃ(i, j, k, scheme::WENO5S{FT, XT}, ψ, args...) where {FT, XT}
    ψ₂, ψ₁, ψ₀ = left_stencil_x(i, j, k, ψ)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    return w₀ * left_biased_p₀(scheme, ψ₀, XT, Val(1), args...) + 
           w₁ * left_biased_p₁(scheme, ψ₁, XT, Val(1), args...) + 
           w₂ * left_biased_p₂(scheme, ψ₂, XT, Val(1), args...)
end

@inline function weno_left_biased_interpolate_yᵃᶠᵃ(i, j, k, scheme::WENO5S{FT, XT, YT}, ψ, args...) where {FT, XT, YT}
    ψ₂, ψ₁, ψ₀ = left_stencil_y(i, j, k, ψ)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    return w₀ * left_biased_p₀(scheme, ψ₀, YT, Val(2), args...) + 
           w₁ * left_biased_p₁(scheme, ψ₁, YT, Val(2), args...) + 
           w₂ * left_biased_p₂(scheme, ψ₂, YT, Val(2), args...)
end

@inline function weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k, scheme::WENO5S{FT, XT, YT, ZT}, ψ, args...) where {FT, XT, YT, ZT}
    ψ₂, ψ₁, ψ₀ = left_stencil_z(i, j, k, ψ)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    return w₀ * left_biased_p₀(scheme, ψ₀, ZT, Val(3), args...) +
           w₁ * left_biased_p₁(scheme, ψ₁, ZT, Val(3), args...) + 
           w₂ * left_biased_p₂(scheme, ψ₂, ZT, Val(3), args...)
end

@inline function weno_right_biased_interpolate_xᶠᵃᵃ(i, j, k, scheme::WENO5S{FT, XT}, ψ, args...) where {FT, XT}
    ψ₂, ψ₁, ψ₀ = right_stencil_x(i, j, k, ψ)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    return w₀ * right_biased_p₀(scheme, ψ₀, XT, Val(1), args...) +
           w₁ * right_biased_p₁(scheme, ψ₁, XT, Val(1), args...) +
           w₂ * right_biased_p₂(scheme, ψ₂, XT, Val(1), args...)
end

@inline function weno_right_biased_interpolate_yᵃᶠᵃ(i, j, k, scheme::WENO5S{FT, XT, YT}, ψ, args...) where {FT, XT, YT}
    ψ₂, ψ₁, ψ₀ = right_stencil_y(i, j, k, ψ)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    return w₀ * right_biased_p₀(scheme, ψ₀, YT, Val(2), args...) + 
           w₁ * right_biased_p₁(scheme, ψ₁, YT, Val(2), args...) + 
           w₂ * right_biased_p₂(scheme, ψ₂, YT, Val(2), args...)
end

@inline function weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k,scheme::WENO5S{FT, XT, YT, ZT}, ψ, args...) where {FT, XT, YT, ZT}
    ψ₂, ψ₁, ψ₀ = right_stencil_z(i, j, k, ψ)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀)
    return w₀ * right_biased_p₀(scheme, ψ₀, ZT, Val(3), args...) + 
           w₁ * right_biased_p₁(scheme, ψ₁, ZT, Val(3), args...) +
           w₂ * right_biased_p₂(scheme, ψ₂, ZT, Val(3), args...)
end

#####
##### Coefficients for stretched (and uniform) ENO schemes (see Shu )
#####

@inline coeff_left_p₀(scheme::WENO5S{FT}, ::Type{Nothing}, args...) where FT = (  FT(1/3),    FT(5/6), - FT(1/6))
@inline coeff_left_p₁(scheme::WENO5S{FT}, ::Type{Nothing}, args...) where FT = (- FT(1/6),    FT(5/6),   FT(1/3))
@inline coeff_left_p₂(scheme::WENO5S{FT}, ::Type{Nothing}, args...) where FT = (  FT(1/3),  - FT(7/6),  FT(11/6))

@inline coeff_right_p₀(scheme, ::Type{Nothing}, args...) = reverse(coeff_left_p₂(scheme, Nothing, args...)) 
@inline coeff_right_p₁(scheme, ::Type{Nothing}, args...) = reverse(coeff_left_p₁(scheme, Nothing, args...)) 
@inline coeff_right_p₂(scheme, ::Type{Nothing}, args...) = reverse(coeff_left_p₀(scheme, Nothing, args...)) 

@inline coeff_left_p₀(scheme, T, dir, i, loc) = retrieve_coeff(scheme, 0, dir, i ,loc)
@inline coeff_left_p₁(scheme, T, dir, i, loc) = retrieve_coeff(scheme, 1, dir, i ,loc)
@inline coeff_left_p₂(scheme, T, dir, i, loc) = retrieve_coeff(scheme, 2, dir, i ,loc)

@inline coeff_right_p₀(scheme, T, dir, i, loc) = retrieve_coeff(scheme, -1, dir, i ,loc)
@inline coeff_right_p₁(scheme, T, dir, i, loc) = retrieve_coeff(scheme,  0, dir, i ,loc)
@inline coeff_right_p₂(scheme, T, dir, i, loc) = retrieve_coeff(scheme,  1, dir, i ,loc)

@inline retrieve_coeff(scheme, r, ::Val{1}, i, ::Type{Face})   = ( scheme.coeff_xᶠᵃᵃ[r+2][1][i], scheme.coeff_xᶠᵃᵃ[r+2][2][i], scheme.coeff_xᶠᵃᵃ[r+2][3][i] )
@inline retrieve_coeff(scheme, r, ::Val{1}, i, ::Type{Center}) = ( scheme.coeff_xᶜᵃᵃ[r+2][1][i], scheme.coeff_xᶜᵃᵃ[r+2][2][i], scheme.coeff_xᶜᵃᵃ[r+2][3][i] )
@inline retrieve_coeff(scheme, r, ::Val{2}, i, ::Type{Face})   = ( scheme.coeff_yᵃᶠᵃ[r+2][1][i], scheme.coeff_yᵃᶠᵃ[r+2][2][i], scheme.coeff_yᵃᶠᵃ[r+2][3][i] )
@inline retrieve_coeff(scheme, r, ::Val{2}, i, ::Type{Center}) = ( scheme.coeff_yᵃᶜᵃ[r+2][1][i], scheme.coeff_yᵃᶜᵃ[r+2][2][i], scheme.coeff_yᵃᶜᵃ[r+2][3][i] )
@inline retrieve_coeff(scheme, r, ::Val{3}, i, ::Type{Face})   = ( scheme.coeff_zᵃᵃᶠ[r+2][1][i], scheme.coeff_zᵃᵃᶠ[r+2][2][i], scheme.coeff_zᵃᵃᶠ[r+2][3][i] )
@inline retrieve_coeff(scheme, r, ::Val{3}, i, ::Type{Center}) = ( scheme.coeff_zᵃᵃᶜ[r+2][1][i], scheme.coeff_zᵃᵃᶜ[r+2][2][i], scheme.coeff_zᵃᵃᶜ[r+2][3][i] )

function calc_interpolating_coefficients(FT, coord, arch, N) 

    c₋₁    = ( OffsetArray(zeros(FT, length(coord)), coord.offsets[1]),
               OffsetArray(zeros(FT, length(coord)), coord.offsets[1]), 
               OffsetArray(zeros(FT, length(coord)), coord.offsets[1]) )
    
    c₀     = ( OffsetArray(zeros(FT, length(coord)), coord.offsets[1]),
               OffsetArray(zeros(FT, length(coord)), coord.offsets[1]), 
               OffsetArray(zeros(FT, length(coord)), coord.offsets[1]) )

    c₁     = ( OffsetArray(zeros(FT, length(coord)), coord.offsets[1]),
               OffsetArray(zeros(FT, length(coord)), coord.offsets[1]), 
               OffsetArray(zeros(FT, length(coord)), coord.offsets[1]) )

    c₂     = ( OffsetArray(zeros(FT, length(coord)), coord.offsets[1]),
               OffsetArray(zeros(FT, length(coord)), coord.offsets[1]), 
               OffsetArray(zeros(FT, length(coord)), coord.offsets[1]) )

    @inbounds begin
        for j = 0:2
            for i = 0:N+1
                c₋₁[j+1][i] = interpolation_weights(-1, coord, j, i)
                c₀[j+1][i]  = interpolation_weights( 0, coord, j, i)
                c₁[j+1][i]  = interpolation_weights( 1, coord, j, i)
                c₂[j+1][i]  = interpolation_weights( 2, coord, j, i)
            end
        end
    end

    c₋₁ = ( OffsetArray(arch_array(arch, parent(c₋₁[1])), coord.offsets[1]),
            OffsetArray(arch_array(arch, parent(c₋₁[2])), coord.offsets[1]),
            OffsetArray(arch_array(arch, parent(c₋₁[3])), coord.offsets[1]) )
    c₀  = ( OffsetArray(arch_array(arch, parent(c₀[1])) , coord.offsets[1]),
            OffsetArray(arch_array(arch, parent(c₀[2])) , coord.offsets[1]),
            OffsetArray(arch_array(arch, parent(c₀[3])) , coord.offsets[1]) )
    c₁  = ( OffsetArray(arch_array(arch, parent(c₁[1])) , coord.offsets[1]),
            OffsetArray(arch_array(arch, parent(c₁[2])) , coord.offsets[1]),
            OffsetArray(arch_array(arch, parent(c₁[3])) , coord.offsets[1]) )
    c₂  = ( OffsetArray(arch_array(arch, parent(c₂[1])) , coord.offsets[1]),
            OffsetArray(arch_array(arch, parent(c₂[2])) , coord.offsets[1]),
            OffsetArray(arch_array(arch, parent(c₂[3])) , coord.offsets[1]) )

    return (c₋₁, c₀, c₁, c₂)
end

function interpolation_weights(r, coord, j, i)

    c = 0
    @inbounds begin
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
    end
    return c * (coord[i-r+j] - coord[i-r+j-1])
end