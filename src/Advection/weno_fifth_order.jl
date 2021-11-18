#####
##### Weighted Essentially Non-Oscillatory (WENO) scheme of order 5
#####

using OffsetArrays
using Oceananigans.Grids: with_halo
using Oceananigans.Architectures: arch_array
using Adapt
import Base: show

const two_32 = Int32(2)

const C3₀ = 3/10
const C3₁ = 3/5
const C3₂ = 1/10

const ƞ = Int32(2) # WENO exponent
const ε = 1e-6

struct WENO5{FT, XT, YT, ZT, XS, YS, ZS, W} <: AbstractUpwindBiasedAdvectionScheme{2} 
    
    # coefficients for ENO reconstruction 
    coeff_xᶠᵃᵃ::XT
    coeff_xᶜᵃᵃ::XT   
    coeff_yᵃᶠᵃ::YT
    coeff_yᵃᶜᵃ::YT
    coeff_zᵃᵃᶠ::ZT
    coeff_zᵃᵃᶜ::ZT
    # coefficients to calculate WENO smoothness indicators
    smooth_xᶠᵃᵃ::XS
    smooth_xᶜᵃᵃ::XS
    smooth_yᵃᶠᵃ::YS
    smooth_yᵃᶜᵃ::YS
    smooth_zᵃᵃᶠ::ZS
    smooth_zᵃᵃᶜ::ZS
end

function WENO5(FT = Float64; grid = nothing, stretched_smoothness = false, zweno = false) 
    
    metrics   = (:xᶠᵃᵃ, :xᶜᵃᵃ, :yᵃᶠᵃ, :yᵃᶜᵃ, :zᵃᵃᶠ, :zᵃᵃᶜ)
    dirsize   = (:Nx, :Nx, :Ny, :Ny, :Nz, :Nz)

    if grid isa Nothing
        @warn "defaulting to uniform WENO scheme with $(FT) precision, use WENO5(grid = grid) if this was not intended"
        for metric in metrics
            @eval $(Symbol(:coeff_ , metric)) = nothing
            @eval $(Symbol(:smooth_, metric)) = nothing
        end
    elseif !(grid isa RectilinearGrid)
        FT = Float32
        @warn "Stretched WENO is not supported with grids other than Rectilinear, defaulting to Uniform WENO"
        for metric in metrics
            @eval $(Symbol(:coeff_ , metric)) = nothing
            @eval $(Symbol(:smooth_, metric)) = nothing
        end
    else
        FT       = eltype(grid)
        arch     = grid.architecture
        new_grid = with_halo((4,4,4), grid)
       
        for (dir, metric) in zip(dirsize, metrics)
            @eval $(Symbol(:coeff_ , metric)) = calc_interpolating_coefficients($FT, $new_grid.$metric, $arch, $new_grid.$dir)
            @eval $(Symbol(:smooth_, metric)) = calc_smoothness_coefficients($FT, $Val($stretched_smoothness), $new_grid.$metric, $arch, $new_grid.$dir) 
        end
    end

    XT = typeof(coeff_xᶠᵃᵃ)
    YT = typeof(coeff_yᵃᶠᵃ)
    ZT = typeof(coeff_zᵃᵃᶠ)
    XS = typeof(smooth_xᶠᵃᵃ)
    YS = typeof(smooth_yᵃᶠᵃ)
    ZS = typeof(smooth_zᵃᵃᶠ)
    zweno ? W  = Number : W = Nothing

    return WENO5{FT, XT, YT, ZT, XS, YS, ZS, W}(coeff_xᶠᵃᵃ , coeff_xᶜᵃᵃ , coeff_yᵃᶠᵃ , coeff_yᵃᶜᵃ , coeff_zᵃᵃᶠ , coeff_zᵃᵃᶜ ,
                                                smooth_xᶠᵃᵃ, smooth_xᶜᵃᵃ, smooth_yᵃᶠᵃ, smooth_yᵃᶜᵃ, smooth_zᵃᵃᶠ, smooth_zᵃᵃᶜ)
end

const JSWENO = WENO5{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Nothing}
const ZWENO  = WENO5{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}

function Base.show(io::IO, a::WENO5{FT, RX, RY, RZ}) where {FT, RX, RY, RZ}
    print(io, "WENO5 advection sheme with X $(RX == Nothing ? "regular" : "stretched") \n",
                                        " Y $(RY == Nothing ? "regular" : "stretched") \n",
                                        " Z $(RZ == Nothing ? "regular" : "stretched")" )
end

Adapt.adapt_structure(to, scheme::WENO5{FT, XT, YT, ZT, XS, YS, ZS}) where {FT, XT, YT, ZT, XS, YS, ZS} =
     WENO5{FT, typeof(Adapt.adapt(to, scheme.coeff_xᶠᵃᵃ)),
               typeof(Adapt.adapt(to, scheme.coeff_yᵃᶠᵃ)),  
               typeof(Adapt.adapt(to, scheme.coeff_zᵃᵃᶠ)),
               typeof(Adapt.adapt(to, scheme.smooth_xᶠᵃᵃ)),
               typeof(Adapt.adapt(to, scheme.smooth_yᵃᶠᵃ)),  
               typeof(Adapt.adapt(to, scheme.smooth_zᵃᵃᶠ))}(
        Adapt.adapt(to, scheme.coeff_xᶠᵃᵃ),
        Adapt.adapt(to, scheme.coeff_xᶜᵃᵃ),
        Adapt.adapt(to, scheme.coeff_yᵃᶠᵃ),
        Adapt.adapt(to, scheme.coeff_yᵃᶜᵃ),
        Adapt.adapt(to, scheme.coeff_zᵃᵃᶠ),       
        Adapt.adapt(to, scheme.coeff_zᵃᵃᶜ),
        Adapt.adapt(to, scheme.smooth_xᶠᵃᵃ),
        Adapt.adapt(to, scheme.smooth_xᶜᵃᵃ),
        Adapt.adapt(to, scheme.smooth_yᵃᶠᵃ),
        Adapt.adapt(to, scheme.smooth_yᵃᶜᵃ),
        Adapt.adapt(to, scheme.smooth_zᵃᵃᶠ),       
        Adapt.adapt(to, scheme.smooth_zᵃᵃᶜ))

@inline boundary_buffer(::WENO5) = 2

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO5, c) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO5, c) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, centered_fourth_order, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO5, c) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, centered_fourth_order, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::WENO5, u) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, centered_fourth_order, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::WENO5, v) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, centered_fourth_order, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::WENO5, w) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, centered_fourth_order, w)

# Unroll the functions to pass the coordinates in case of a stretched grid
@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO5, ψ)  = weno_left_biased_interpolate_xᶠᵃᵃ(i, j, k, scheme, ψ, i, Face)
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO5, ψ)  = weno_left_biased_interpolate_yᵃᶠᵃ(i, j, k, scheme, ψ, j, Face)
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO5, ψ)  = weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k, scheme, ψ, k, Face)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO5, ψ) = weno_right_biased_interpolate_xᶠᵃᵃ(i, j, k, scheme, ψ, i, Face)
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO5, ψ) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j, k, scheme, ψ, j, Face)
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO5, ψ) = weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k, scheme, ψ, k, Face)

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5, ψ)  = weno_left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, scheme, ψ, i, Center)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5, ψ)  = weno_left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, scheme, ψ, j, Center)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5, ψ)  = weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, scheme, ψ, k, Center)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO5, ψ) = weno_right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, scheme, ψ, i, Center)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO5, ψ) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, scheme, ψ, j, Center)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO5, ψ) = weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, scheme, ψ, k, Center)

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


@inline left_biased_β₀(FT, ψ, ::Type{Nothing}, args...) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32
@inline left_biased_β₁(FT, ψ, ::Type{Nothing}, args...) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * ( ψ[1]         -  ψ[3])^two_32
@inline left_biased_β₂(FT, ψ, ::Type{Nothing}, args...) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32

@inline right_biased_β₀(FT, ψ, ::Type{Nothing}, args...) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * ( ψ[1] - 4ψ[2] + 3ψ[3])^two_32
@inline right_biased_β₁(FT, ψ, ::Type{Nothing}, args...) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * ( ψ[1]         -  ψ[3])^two_32
@inline right_biased_β₂(FT, ψ, ::Type{Nothing}, args...) = @inbounds FT(13/12) * (ψ[1] - 2ψ[2] + ψ[3])^two_32 + FT(1/4) * (3ψ[1] - 4ψ[2] +  ψ[3])^two_32

#####
##### Stretched smoothness indicators gathered from precomputed values.
##### the stretched values for β coefficients is calculated from 
##### Shu, NASA/CR-97-206253, ICASE Report No. 97-65
##### by hardcoding that p(x) is a 2nd order polynomial
#####


@inline function biased_left_β(ψ, scheme, r, args...) 
    stencil   = retrieve_left_smooth(scheme, r, args...)
    wᵢᵢ = stencil[1]   
    wᵢⱼ = stencil[2]
    
    return   sum(ψ .* ( wᵢᵢ .* ψ .+ wᵢⱼ .* dagger(ψ) ) )
end

@inline function biased_right_β(ψ, scheme, r, args...) 
    stencil   = retrieve_right_smooth(scheme, r, args...)
    wᵢᵢ = stencil[1]   
    wᵢⱼ = stencil[2]
    
    return   sum(ψ .* ( wᵢᵢ .* ψ .+ wᵢⱼ .* dagger(ψ) ) )
end

@inline left_biased_β₀(FT, ψ, T, scheme, args...) = @inbounds biased_left_β(ψ, scheme, 0, args...) 
@inline left_biased_β₁(FT, ψ, T, scheme, args...) = @inbounds biased_left_β(ψ, scheme, 1, args...) 
@inline left_biased_β₂(FT, ψ, T, scheme, args...) = @inbounds biased_left_β(ψ, scheme, 2, args...) 

@inline right_biased_β₀(FT, ψ, T, scheme, args...) = @inbounds biased_right_β(ψ, scheme, 2, args...) 
@inline right_biased_β₁(FT, ψ, T, scheme, args...) = @inbounds biased_right_β(ψ, scheme, 1, args...) 
@inline right_biased_β₂(FT, ψ, T, scheme, args...) = @inbounds biased_right_β(ψ, scheme, 0, args...) 

# Right-biased smoothness indicators are a reflection or "symmetric modification" of the left-biased smoothness
# indicators around grid point `i-1/2`.

@inline left_biased_α₀(FT, ψ, args...) = FT(C3₀) / (left_biased_β₀(FT, ψ, args...) + FT(ε))^ƞ
@inline left_biased_α₁(FT, ψ, args...) = FT(C3₁) / (left_biased_β₁(FT, ψ, args...) + FT(ε))^ƞ
@inline left_biased_α₂(FT, ψ, args...) = FT(C3₂) / (left_biased_β₂(FT, ψ, args...) + FT(ε))^ƞ

@inline right_biased_α₀(FT, ψ, args...) = FT(C3₂) / (right_biased_β₀(FT, ψ, args...) + FT(ε))^ƞ
@inline right_biased_α₁(FT, ψ, args...) = FT(C3₁) / (right_biased_β₁(FT, ψ, args...) + FT(ε))^ƞ
@inline right_biased_α₂(FT, ψ, args...) = FT(C3₀) / (right_biased_β₂(FT, ψ, args...) + FT(ε))^ƞ

#####
##### Z-WENO-5 reconstruction (Castro et al: High order weighted essentially non-oscillatory WENO-Z schemesfor hyperbolic conservation laws)
#####

@inline function left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀, T, scheme::ZWENO, args...)
    β₀ = left_biased_β₀(FT, ψ₀, T, scheme, args...)
    β₁ = left_biased_β₁(FT, ψ₁, T, scheme, args...)
    β₂ = left_biased_β₂(FT, ψ₂, T, scheme, args...)
    
    τ₅ = abs(β₂ - β₀)
    α₀ = FT(C3₀) * (1 + (τ₅ / (β₀ + FT(ε)))^ƞ) 
    α₁ = FT(C3₁) * (1 + (τ₅ / (β₁ + FT(ε)))^ƞ) 
    α₂ = FT(C3₂) * (1 + (τ₅ / (β₂ + FT(ε)))^ƞ) 

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀, T, scheme::ZWENO, args...)
    β₀ = right_biased_β₀(FT, ψ₀, T, scheme, args...)
    β₁ = right_biased_β₁(FT, ψ₁, T, scheme, args...)
    β₂ = right_biased_β₂(FT, ψ₂, T, scheme, args...)

    τ₅ = abs(β₂ - β₀)
    α₀ = FT(C3₂) * (1 + (τ₅ / (β₀ + FT(ε)))^ƞ) 
    α₁ = FT(C3₁) * (1 + (τ₅ / (β₁ + FT(ε)))^ƞ) 
    α₂ = FT(C3₀) * (1 + (τ₅ / (β₂ + FT(ε)))^ƞ) 
    
    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

#####
##### JS-WENO-5 reconstruction
#####

@inline function left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀, T, scheme::JSWENO, args...)
    α₀ = left_biased_α₀(FT, ψ₀, T, scheme, args...)
    α₁ = left_biased_α₁(FT, ψ₁, T, scheme, args...)
    α₂ = left_biased_α₂(FT, ψ₂, T, scheme, args...)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

@inline function right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀, T, scheme::JSWENO, args...)
    α₀ = right_biased_α₀(FT, ψ₀, T, scheme, args...)
    α₁ = right_biased_α₁(FT, ψ₁, T, scheme, args...)
    α₂ = right_biased_α₂(FT, ψ₂, T, scheme, args...)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

#####
##### Biased interpolation functions
#####

@inline function weno_left_biased_interpolate_xᶠᵃᵃ(i, j, k, scheme::WENO5{FT, XT, YT, ZT, XS, YS, ZS}, ψ, args...) where {FT, XT, YT, ZT, XS, YS, ZS}
    ψ₂, ψ₁, ψ₀ = left_stencil_x(i, j, k, ψ)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀, XS, scheme, Val(1), args...)
    return w₀ * left_biased_p₀(scheme, ψ₀, XT, Val(1), args...) + 
           w₁ * left_biased_p₁(scheme, ψ₁, XT, Val(1), args...) + 
           w₂ * left_biased_p₂(scheme, ψ₂, XT, Val(1), args...)
end

@inline function weno_left_biased_interpolate_yᵃᶠᵃ(i, j, k, scheme::WENO5{FT, XT, YT, ZT, XS, YS, ZS}, ψ, args...) where {FT, XT, YT, ZT, XS, YS, ZS}
    ψ₂, ψ₁, ψ₀ = left_stencil_y(i, j, k, ψ)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀, YS, scheme, Val(2), args...)
    return w₀ * left_biased_p₀(scheme, ψ₀, YT, Val(2), args...) + 
           w₁ * left_biased_p₁(scheme, ψ₁, YT, Val(2), args...) + 
           w₂ * left_biased_p₂(scheme, ψ₂, YT, Val(2), args...)
end

@inline function weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k, scheme::WENO5{FT, XT, YT, ZT, XS, YS, ZS}, ψ, args...) where {FT, XT, YT, ZT, XS, YS, ZS}
    ψ₂, ψ₁, ψ₀ = left_stencil_z(i, j, k, ψ)
    w₀, w₁, w₂ = left_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀, ZS, scheme, Val(3), args...)
    return w₀ * left_biased_p₀(scheme, ψ₀, ZT, Val(3), args...) +
           w₁ * left_biased_p₁(scheme, ψ₁, ZT, Val(3), args...) + 
           w₂ * left_biased_p₂(scheme, ψ₂, ZT, Val(3), args...)
end

@inline function weno_right_biased_interpolate_xᶠᵃᵃ(i, j, k, scheme::WENO5{FT, XT, YT, ZT, XS, YS, ZS}, ψ, args...) where {FT, XT, YT, ZT, XS, YS, ZS}
    ψ₂, ψ₁, ψ₀ = right_stencil_x(i, j, k, ψ)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀, XS, scheme, Val(1), args...)
    return w₀ * right_biased_p₀(scheme, ψ₀, XT, Val(1), args...) +
           w₁ * right_biased_p₁(scheme, ψ₁, XT, Val(1), args...) +
           w₂ * right_biased_p₂(scheme, ψ₂, XT, Val(1), args...)
end

@inline function weno_right_biased_interpolate_yᵃᶠᵃ(i, j, k, scheme::WENO5{FT, XT, YT, ZT, XS, YS, ZS}, ψ, args...) where {FT, XT, YT, ZT, XS, YS, ZS}
    ψ₂, ψ₁, ψ₀ = right_stencil_y(i, j, k, ψ)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀, YS, scheme, Val(2), args...)
    return w₀ * right_biased_p₀(scheme, ψ₀, YT, Val(2), args...) + 
           w₁ * right_biased_p₁(scheme, ψ₁, YT, Val(2), args...) + 
           w₂ * right_biased_p₂(scheme, ψ₂, YT, Val(2), args...)
end

@inline function weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k,scheme::WENO5{FT, XT, YT, ZT, XS, YS, ZS}, ψ, args...) where {FT, XT, YT, ZT, XS, YS, ZS}
    ψ₂, ψ₁, ψ₀ = right_stencil_z(i, j, k, ψ)
    w₀, w₁, w₂ = right_biased_weno5_weights(FT, ψ₂, ψ₁, ψ₀, ZS, scheme, Val(3), args...)
    return w₀ * right_biased_p₀(scheme, ψ₀, ZT, Val(3), args...) + 
           w₁ * right_biased_p₁(scheme, ψ₁, ZT, Val(3), args...) +
           w₂ * right_biased_p₂(scheme, ψ₂, ZT, Val(3), args...)
end

#####
##### Coefficients for stretched (and uniform) ENO schemes (see Shu NASA/CR-97-206253, ICASE Report No. 97-65)
#####

@inline coeff_left_p₀(scheme::WENO5{FT}, ::Type{Nothing}, args...) where FT = (  FT(1/3),    FT(5/6), - FT(1/6))
@inline coeff_left_p₁(scheme::WENO5{FT}, ::Type{Nothing}, args...) where FT = (- FT(1/6),    FT(5/6),   FT(1/3))
@inline coeff_left_p₂(scheme::WENO5{FT}, ::Type{Nothing}, args...) where FT = (  FT(1/3),  - FT(7/6),  FT(11/6))

@inline coeff_right_p₀(scheme, ::Type{Nothing}, args...) = reverse(coeff_left_p₂(scheme, Nothing, args...)) 
@inline coeff_right_p₁(scheme, ::Type{Nothing}, args...) = reverse(coeff_left_p₁(scheme, Nothing, args...)) 
@inline coeff_right_p₂(scheme, ::Type{Nothing}, args...) = reverse(coeff_left_p₀(scheme, Nothing, args...)) 

@inline coeff_left_p₀(scheme, T, dir, i, loc) = retrieve_coeff(scheme, 0, dir, i ,loc)
@inline coeff_left_p₁(scheme, T, dir, i, loc) = retrieve_coeff(scheme, 1, dir, i ,loc)
@inline coeff_left_p₂(scheme, T, dir, i, loc) = retrieve_coeff(scheme, 2, dir, i ,loc)

@inline coeff_right_p₀(scheme, T, dir, i, loc) = retrieve_coeff(scheme, -1, dir, i ,loc)
@inline coeff_right_p₁(scheme, T, dir, i, loc) = retrieve_coeff(scheme,  0, dir, i ,loc)
@inline coeff_right_p₂(scheme, T, dir, i, loc) = retrieve_coeff(scheme,  1, dir, i ,loc)

@inline retrieve_coeff(scheme, r, ::Val{1}, i, ::Type{Face})   = scheme.coeff_xᶠᵃᵃ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{1}, i, ::Type{Center}) = scheme.coeff_xᶜᵃᵃ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{2}, i, ::Type{Face})   = scheme.coeff_yᵃᶠᵃ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{2}, i, ::Type{Center}) = scheme.coeff_yᵃᶜᵃ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{3}, i, ::Type{Face})   = scheme.coeff_zᵃᵃᶠ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{3}, i, ::Type{Center}) = scheme.coeff_zᵃᵃᶜ[r+2][i] 

@inline retrieve_left_smooth(scheme, r, ::Val{1}, i, ::Type{Face})   = scheme.smooth_xᶠᵃᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{1}, i, ::Type{Center}) = scheme.smooth_xᶜᵃᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{2}, i, ::Type{Face})   = scheme.smooth_yᵃᶠᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{2}, i, ::Type{Center}) = scheme.smooth_yᵃᶜᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{3}, i, ::Type{Face})   = scheme.smooth_zᵃᵃᶠ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{3}, i, ::Type{Center}) = scheme.smooth_zᵃᵃᶜ[r+1][i] 

@inline retrieve_right_smooth(scheme, r, ::Val{1}, i, ::Type{Face})   = scheme.smooth_xᶠᵃᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{1}, i, ::Type{Center}) = scheme.smooth_xᶜᵃᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{2}, i, ::Type{Face})   = scheme.smooth_yᵃᶠᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{2}, i, ::Type{Center}) = scheme.smooth_yᵃᶜᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{3}, i, ::Type{Face})   = scheme.smooth_zᵃᵃᶠ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{3}, i, ::Type{Center}) = scheme.smooth_zᵃᵃᶜ[r+4][i] 


@inline calc_interpolating_coefficients(FT, coord::OffsetArray{<:Any, <:Any, <:AbstractRange}, arch, N) = nothing
@inline calc_interpolating_coefficients(FT, coord::AbstractRange, arch, N)                              = nothing

@inline calc_smoothness_coefficients(FT, ::Val{false}, args...) = nothing
@inline calc_smoothness_coefficients(FT, ::Val{true}, coord::OffsetArray{<:Any, <:Any, <:AbstractRange}, arch, N) = nothing
@inline calc_smoothness_coefficients(FT, ::Val{true}, coord::AbstractRange, arch, N) = nothing


function calc_interpolating_coefficients(FT, coord, arch, N) 

    cpu_coord = Array(parent(coord))
    cpu_coord = OffsetArray(cpu_coord, coord.offsets[1])

    allstencils = ()
    for r = -1:2
        stencil = NTuple{3, FT}[]
        @inbounds begin
            for i = 0:N+1
                push!(stencil, interp_weights(r, cpu_coord, i, 0, -))     
            end
        end
        stencil     = OffsetArray(arch_array(arch, stencil), -1)
        allstencils = (allstencils..., stencil)
    end

    return allstencils
end

function calc_smoothness_coefficients(FT, beta, coord, arch, N) 

    cpu_coord = Array(parent(coord))
    cpu_coord = OffsetArray(cpu_coord, coord.offsets[1])

    # derivation written on overleaf

    allstencils = ()
    for op = (-, +)
        for r = 0:2
            
            stencil = NTuple{2, NTuple{3, FT}}[]   
            @inbounds begin
                for i = 0:N+1
               
                    bias1 = Int(op == +)
                    bias2 = bias1 - 1
    
                    Δcᵢ = cpu_coord[i + bias1] - cpu_coord[i + bias2]
                
                    Bᵢ  = prim_interp_weights(r, cpu_coord, i, bias1, op)
                    bᵢ  =      interp_weights(r, cpu_coord, i, bias1, op)
                    bₓᵢ = der1_interp_weights(r, cpu_coord, i, bias1, op)
                    Aᵢ  = prim_interp_weights(r, cpu_coord, i, bias2, op)
                    aᵢ  =      interp_weights(r, cpu_coord, i, bias2, op)
                    aₓᵢ = der1_interp_weights(r, cpu_coord, i, bias2, op)
    
                    pₓₓ = der2_interp_weights(r, cpu_coord, i, op)
                    Pᵢ  =  (Bᵢ .- Aᵢ)
    
                    wᵢᵢ = Δcᵢ  .* (bᵢ .* bₓᵢ .- aᵢ .* aₓᵢ .- pₓₓ .* Pᵢ)  .+ Δcᵢ^4 .* (pₓₓ .* pₓₓ)
                    wᵢⱼ = Δcᵢ  .* (star(bᵢ, bₓᵢ)  .- star(aᵢ, aₓᵢ) .- star(pₓₓ, Pᵢ)) .+
                                                         Δcᵢ^4 .* star(pₓₓ, pₓₓ)
    
                    push!(stencil, (wᵢᵢ, wᵢⱼ))
                end
            end
    
            stencil     = OffsetArray(arch_array(arch, stencil), -1)
            allstencils = (allstencils..., stencil)
        end
    end

    return allstencils
end

@inline dagger(ψ)    = (ψ[2:3]..., ψ[1])
@inline star(ψ₁, ψ₂) = (ψ₁ .* dagger(ψ₂) .+ dagger(ψ₁) .* ψ₂)

# Integral of ENO coefficients for 2nd order polynomial reconstruction at the face
function prim_interp_weights(r, coord, i, bias, op)

    coeff = ()
    for j = 0:2
        c = 0
        @inbounds begin
            for m = j+1:3
                num = 0
                for l = 0:3
                    if l != m
                        prod = 1
                        sum  = 0 
                        for q = 0:3
                            if q != m && q != l 
                                prod *= coord[op(i, r-q+1)]
                                sum  += coord[op(i, r-q+1)]
                            end
                        end
                        num += coord[i+bias]^3 / 3 - sum * coord[i+bias]^2 / 2 + prod * coord[i+bias]
                    end
                end
                den = 1
                for l = 0:3
                    if l!= m
                        den *= (coord[op(i, r-m+1)] - coord[op(i, r-l+1)])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[op(i, r-j)] - coord[op(i, r-j+1)]))
    end

    return coeff
end

# Second derivative of ENO coefficients for 2nd order polynomial reconstruction at the face
function der2_interp_weights(r, coord, i, op)

    coeff = ()
    for j = 0:2
        c = 0
        @inbounds begin
            for m = j+1:3
                num = 0
                for l = 0:3
                    if l != m
                        num += 2 
                    end
                end
                den = 1
                for l = 0:3
                    if l!= m
                        den *= (coord[op(i, r-m+1)] - coord[op(i, r-l+1)])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[op(i, r-j)] - coord[op(i, r-j+1)]))
    end

    return coeff
end

# first derivative of ENO coefficients for 2nd order polynomial reconstruction at the face
function der1_interp_weights(r, coord, i, bias, op)

    coeff = ()
    for j = 0:2
        c = 0
        @inbounds begin
            for m = j+1:3
                num = 0
                for l = 0:3
                    if l != m
                        sum = 0
                        for q = 0:3
                            if q != m && q != l 
                                sum += coord[op(i, r-q+1)]
                            end
                        end
                        num += 2 * coord[i+bias] - sum
                    end
                end
                den = 1
                for l = 0:3
                    if l!= m
                        den *= (coord[op(i, r-m+1)] - coord[op(i, r-l+1)])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[op(i, r-j)] - coord[op(i, r-j+1)]))
    end

    return coeff
end

# ENO coefficients for 2nd order polynomial reconstruction at the face
function interp_weights(r, coord, i, bias, op)

    coeff = ()
    for j = 0:2
        c = 0
        @inbounds begin
            for m = j+1:3
                num = 0
                for l = 0:3
                    if l != m
                        prod = 1
                        for q = 0:3
                            if q != m && q != l 
                                prod *= (coord[i+bias] - coord[op(i, r-q+1)])
                            end
                        end
                        num += prod
                    end
                end
                den = 1
                for l = 0:3
                    if l!= m
                        den *= (coord[op(i, r-m+1)] - coord[op(i, r-l+1)])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[op(i, r-j)] - coord[op(i, r-j+1)]))
    end

    return coeff
end



