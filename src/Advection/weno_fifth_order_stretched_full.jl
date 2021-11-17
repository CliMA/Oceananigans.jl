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

struct WENO5{FT, XT, YT, ZT, XS, YS, ZS} <: AbstractUpwindBiasedAdvectionScheme{2} 
    
    coeff_xᶠᵃᵃ::XT
    coeff_xᶜᵃᵃ::XT   
    coeff_yᵃᶠᵃ::YT
    coeff_yᵃᶜᵃ::YT
    coeff_zᵃᵃᶠ::ZT
    coeff_zᵃᵃᶜ::ZT

    smooth_xᶠᵃᵃ::XS
    smooth_xᶜᵃᵃ::XS
    smooth_yᵃᶠᵃ::YS
    smooth_yᵃᶜᵃ::YS
    smooth_zᵃᵃᶠ::ZS
    smooth_zᵃᵃᶜ::ZS

end

function WENO5(FT = Float64; grid = nothing, stretched_smoothness = false) 
    
    if grid isa Nothing
        coeff_xᶠᵃᵃ = nothing 
        coeff_xᶜᵃᵃ = nothing  
        coeff_yᵃᶠᵃ = nothing   
        coeff_yᵃᶜᵃ = nothing
        coeff_zᵃᵃᶠ = nothing
        coeff_zᵃᵃᶜ = nothing
        smooth_xᶠᵃᵃ = nothing 
        smooth_xᶜᵃᵃ = nothing  
        smooth_yᵃᶠᵃ = nothing   
        smooth_yᵃᶜᵃ = nothing
        smooth_zᵃᵃᶠ = nothing
        smooth_zᵃᵃᶜ = nothing        
    elseif !(grid isa RectilinearGrid)
        @warn "Stretched WENO is not supported with grids other than Rectilinear, defaulting to Uniform WENO"
        coeff_xᶠᵃᵃ = nothing 
        coeff_xᶜᵃᵃ = nothing  
        coeff_yᵃᶠᵃ = nothing   
        coeff_yᵃᶜᵃ = nothing
        coeff_zᵃᵃᶠ = nothing
        coeff_zᵃᵃᶜ = nothing
        smooth_xᶠᵃᵃ = nothing 
        smooth_xᶜᵃᵃ = nothing  
        smooth_yᵃᶠᵃ = nothing   
        smooth_yᵃᶜᵃ = nothing
        smooth_zᵃᵃᶠ = nothing
        smooth_zᵃᵃᶜ = nothing        
    else
        FT          = eltype(grid)
        arch        = grid.architecture
        helper_grid = with_halo((4,4,4), grid)
        coeff_xᶠᵃᵃ = calc_interpolating_coefficients(FT, helper_grid.xᶠᵃᵃ, arch, grid.Nx) 
        coeff_xᶜᵃᵃ = calc_interpolating_coefficients(FT, helper_grid.xᶜᵃᵃ, arch, grid.Nx)
        coeff_yᵃᶠᵃ = calc_interpolating_coefficients(FT, helper_grid.yᵃᶠᵃ, arch, grid.Ny)
        coeff_yᵃᶜᵃ = calc_interpolating_coefficients(FT, helper_grid.yᵃᶜᵃ, arch, grid.Ny)
        coeff_zᵃᵃᶠ = calc_interpolating_coefficients(FT, helper_grid.zᵃᵃᶠ, arch, grid.Nz)
        coeff_zᵃᵃᶜ = calc_interpolating_coefficients(FT, helper_grid.zᵃᵃᶜ, arch, grid.Nz)

        smooth_xᶠᵃᵃ = calc_smoothness_coefficients(FT, Val(stretched_smoothness), helper_grid.xᶠᵃᵃ, arch, grid.Nx) 
        smooth_xᶜᵃᵃ = calc_smoothness_coefficients(FT, Val(stretched_smoothness), helper_grid.xᶜᵃᵃ, arch, grid.Nx)
        smooth_yᵃᶠᵃ = calc_smoothness_coefficients(FT, Val(stretched_smoothness), helper_grid.yᵃᶠᵃ, arch, grid.Ny)
        smooth_yᵃᶜᵃ = calc_smoothness_coefficients(FT, Val(stretched_smoothness), helper_grid.yᵃᶜᵃ, arch, grid.Ny)
        smooth_zᵃᵃᶠ = calc_smoothness_coefficients(FT, Val(stretched_smoothness), helper_grid.zᵃᵃᶠ, arch, grid.Nz)
        smooth_zᵃᵃᶜ = calc_smoothness_coefficients(FT, Val(stretched_smoothness), helper_grid.zᵃᵃᶜ, arch, grid.Nz)
    end
    XT = typeof(coeff_xᶠᵃᵃ)
    YT = typeof(coeff_yᵃᶠᵃ)
    ZT = typeof(coeff_zᵃᵃᶠ)
    XS = typeof(smooth_xᶠᵃᵃ)
    YS = typeof(smooth_yᵃᶠᵃ)
    ZS = typeof(smooth_zᵃᵃᶠ)


    return WENO5{FT, XT, YT, ZT, XS, YS, ZS}(coeff_xᶠᵃᵃ , coeff_xᶜᵃᵃ , coeff_yᵃᶠᵃ , coeff_yᵃᶜᵃ , coeff_zᵃᵃᶠ , coeff_zᵃᵃᶜ ,
                                             smooth_xᶠᵃᵃ, smooth_xᶜᵃᵃ, smooth_yᵃᶠᵃ, smooth_yᵃᶜᵃ, smooth_zᵃᵃᶠ, smooth_zᵃᵃᶜ)
end

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

@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO5, ψ) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j, k, scheme, ψ, i, Face)
@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO5, ψ) = weno_right_biased_interpolate_xᶠᵃᵃ(i, j, k, scheme, ψ, j, Face)
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

@inline function biased_β(ψ, scheme, r, args...) 
    stencil   = retrieve_smooth(scheme, r, args...)
    Pᵢ  = stencil[1:3];   
    aᵢ  = stencil[4:6];
    aₓᵢ = stencil[7:9];
    aₓₓ = stencil[10:12];  
    bᵢ  = stencil[13:15];
    bₓᵢ = stencil[16:18];
    bₓₓ = stencil[19:21];
    
    return   dot(aᵢ, ψ) * dot(aₓᵢ, ψ) - dot(bᵢ, ψ) * dot(bₓᵢ, ψ) - dot(aₓₓ, ψ) * dot(Pᵢ,  ψ) + dot(bₓₓ, ψ)
end

@inline left_biased_β₀(FT, ψ, T, scheme, args...) = @inbounds biased_β(ψ, scheme, 0, args...) 
@inline left_biased_β₁(FT, ψ, T, scheme, args...) = @inbounds biased_β(ψ, scheme, 1, args...) 
@inline left_biased_β₂(FT, ψ, T, scheme, args...) = @inbounds biased_β(ψ, scheme, 2, args...) 

@inline right_biased_β₀(FT, ψ, T, scheme, args...) = @inbounds biased_β(ψ, scheme, -1, args...) 
@inline right_biased_β₁(FT, ψ, T, scheme, args...) = @inbounds biased_β(ψ, scheme,  0, args...) 
@inline right_biased_β₂(FT, ψ, T, scheme, args...) = @inbounds biased_β(ψ, scheme,  1, args...) 


# Right-biased smoothness indicators are a reflection or "symmetric modification" of the left-biased smoothness
# indicators around grid point `i-1/2`.

@inline left_biased_α₀(FT, ψ, args...) = FT(C3₀) / (left_biased_β₀(FT, ψ, args...) + FT(ε))^ƞ
@inline left_biased_α₁(FT, ψ, args...) = FT(C3₁) / (left_biased_β₁(FT, ψ, args...) + FT(ε))^ƞ
@inline left_biased_α₂(FT, ψ, args...) = FT(C3₂) / (left_biased_β₂(FT, ψ, args...) + FT(ε))^ƞ

@inline right_biased_α₀(FT, ψ, args...) = FT(C3₂) / (right_biased_β₀(FT, ψ, args...) + FT(ε))^ƞ
@inline right_biased_α₁(FT, ψ, args...) = FT(C3₁) / (right_biased_β₁(FT, ψ, args...) + FT(ε))^ƞ
@inline right_biased_α₂(FT, ψ, args...) = FT(C3₀) / (right_biased_β₂(FT, ψ, args...) + FT(ε))^ƞ

@inline dot(a, b) = sum(a .* b)

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
    α₀ = right_biased_α₀(FT, ψ₀, args...)
    α₁ = right_biased_α₁(FT, ψ₁, args...)
    α₂ = right_biased_α₂(FT, ψ₂, args...)

    Σα = α₀ + α₁ + α₂
    w₀ = α₀ / Σα
    w₁ = α₁ / Σα
    w₂ = α₂ / Σα

    return w₀, w₁, w₂
end

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
##### Coefficients for stretched (and uniform) ENO schemes (see Shu )
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

@inline retrieve_smooth(scheme, r, ::Val{1}, i, ::Type{Face})   = scheme.smooth_xᶠᵃᵃ[r+2][i] 
@inline retrieve_smooth(scheme, r, ::Val{1}, i, ::Type{Center}) = scheme.smooth_xᶜᵃᵃ[r+2][i] 
@inline retrieve_smooth(scheme, r, ::Val{2}, i, ::Type{Face})   = scheme.smooth_yᵃᶠᵃ[r+2][i] 
@inline retrieve_smooth(scheme, r, ::Val{2}, i, ::Type{Center}) = scheme.smooth_yᵃᶜᵃ[r+2][i] 
@inline retrieve_smooth(scheme, r, ::Val{3}, i, ::Type{Face})   = scheme.smooth_zᵃᵃᶠ[r+2][i] 
@inline retrieve_smooth(scheme, r, ::Val{3}, i, ::Type{Center}) = scheme.smooth_zᵃᵃᶜ[r+2][i] 

@inline calc_interpolating_coefficients(FT, coord::OffsetArray{<:Any, <:Any, <:AbstractRange}, arch, N) = nothing
@inline calc_interpolating_coefficients(FT, coord::AbstractRange, arch, N) = nothing

@inline calc_smoothness_coefficients(FT, ::Val{false}, args...) = nothing
@inline calc_smoothness_coefficients(FT, ::Val{true}, coord::OffsetArray{<:Any, <:Any, <:AbstractRange}, arch, N) = nothing
@inline calc_smoothness_coefficients(FT, ::Val{true}, coord::AbstractRange, arch, N) = nothing


function calc_interpolating_coefficients(FT, coord, arch, N) 

    cpu_coord = Array(parent(coord))
    cpu_coord = OffsetArray(cpu_coord, coord.offsets[1])

    c₋₁ = NTuple{3, FT}[]
    c₀  = NTuple{3, FT}[]
    c₁  = NTuple{3, FT}[]
    c₂  = NTuple{3, FT}[]

    @inbounds begin
        for i = 0:N+1
            push!(c₋₁, interp_weights(-1, cpu_coord, i, 0))
            push!(c₀,  interp_weights( 0, cpu_coord, i, 0))
            push!(c₁,  interp_weights( 1, cpu_coord, i, 0))
            push!(c₂,  interp_weights( 2, cpu_coord, i, 0))
        end
    end

    c₋₁ = OffsetArray(arch_array(arch, c₋₁), -1)
    c₀  = OffsetArray(arch_array(arch, c₀ ), -1)
    c₁  = OffsetArray(arch_array(arch, c₁ ), -1)
    c₂  = OffsetArray(arch_array(arch, c₂ ), -1)

    return (c₋₁, c₀, c₁, c₂)
end

function calc_smoothness_coefficients(FT, beta, coord, arch, N) 

    cpu_coord = Array(parent(coord))
    cpu_coord = OffsetArray(cpu_coord, coord.offsets[1])

    ## The smoothness coefficients are calculated as :
    ## Δxᵣ¹ pᵣ(x) * pᵣ'(x) |ₐᵇ - Δxᵣ¹ pᵣ''(x) * Pᵣ(x) |ₐᵇ + pᵣ''(x) * (b - a) Δxᵣ³
    ## where a and b are xᵢ and xᵢ₋₁
    ## and p(x) is the second order reconstruction polynomial while 
    ## P(x), p'(x) and p''(x) are its primitive, first derivative and second derivative, respectively
    ## as p(x) is a second order polynomial, p''(x) is a constant in x

    ## so it's 21 coefficient total (P, p, p' and p'') calculated at xᵢ and xᵢ₋₁ (p''(x) does not depend on x)

    allstencils = ()

    for r = -1:2
        stencil = NTuple{21, FT}[]   

        @inbounds begin
            for i = 0:N+1
                prim  = prim_interp_weights(r, cpu_coord, i, 0)
                val   =      interp_weights(r, cpu_coord, i, 0)
                fir   = der1_interp_weights(r, cpu_coord, i, 0)
                sec   = der2_interp_weights(r, cpu_coord, i, 0)
                primL = prim_interp_weights(r, cpu_coord, i, 1)
                valL  =      interp_weights(r, cpu_coord, i, 1)
                firL  = der1_interp_weights(r, cpu_coord, i, 1)
                
                secI  = der2_integ_interp_weights(r, cpu_coord, i)

                push!(stencil, ((prim .- primL)..., val..., fir..., sec..., valL..., firL..., secI...))
            end
        end

        stencil = OffsetArray(arch_array(arch, stencil), -1)

        allstencils = (allstencils..., stencil)
    end

    return allstencils
end

# Integral of ENO coefficients for 2nd order polynomial reconstruction at the face
function prim_interp_weights(r, coord, i, left)

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
                                prod *= coord[i-r+q-1]
                                sum  += coord[i-r+q-1]
                            end
                        end
                        num += coord[i-left]^3 / 3 - sum * coord[i-left]^2 / 2 + prod * coord[i-left]
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
        coeff = (coeff..., c * (coord[i-r+j] - coord[i-r+j-1]) * (coord[i-r+j] - coord[i-r+j-1]) )
    end

    return coeff
end

# Second derivative of ENO coefficients for 2nd order polynomial reconstruction at the face
function der2_interp_weights(r, coord, i, left)

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
                        den *= (coord[i-r+m-1] - coord[i-r+l-1])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[i-r+j] - coord[i-r+j-1]))
    end

    return coeff
end

# Integrated second derivative of ENO coefficients for 2nd order polynomial reconstruction at the face
function der2_integ_interp_weights(r, coord, i)

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
                        den *= (coord[i-r+m-1] - coord[i-r+l-1])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[i-r+j] - coord[i-r+j-1]) * (coord[i] - coord[i-1]) * (coord[i-r+j] - coord[i-r+j-1])^3)
    end

    return coeff
end

# first derivative of ENO coefficients for 2nd order polynomial reconstruction at the face
function der1_interp_weights(r, coord, i, left)

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
                                sum += coord[i-r+q-1]
                            end
                        end
                        num += 2 * coord[i-left] - sum
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
        coeff = (coeff..., c * (coord[i-r+j] - coord[i-r+j-1]) * (coord[i-r+j] - coord[i-r+j-1]))
    end

    return coeff
end

# ENO coefficients for 2nd order polynomial reconstruction at the face
function interp_weights(r, coord, i, left)

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
                                prod *= (coord[i-left] - coord[i-r+q-1])
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
        coeff = (coeff..., c * (coord[i-r+j] - coord[i-r+j-1]))
    end

    return coeff
end

