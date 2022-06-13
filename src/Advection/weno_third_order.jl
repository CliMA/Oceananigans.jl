#####
##### Weighted Essentially Non-Oscillatory (WENO) third-order advection scheme
#####

const C2₀ = 2/3
const C2₁ = 1/3

const two_32 = Int32(2)

const ƞ = Int32(2) # WENO exponent
const ε = 1e-6

abstract type SmoothnessStencil end

struct VorticityStencil <:SmoothnessStencil end
struct VelocityStencil <:SmoothnessStencil end

"""
    struct WENO3{FT, XT, YT, ZT, XS, YS, ZS, WF} <: AbstractUpwindBiasedAdvectionScheme{3}

Weighted Essentially Non-Oscillatory (WENO) fifth-order advection scheme.

$(TYPEDFIELDS)
"""
struct WENO3{FT, XT, YT, ZT, XS, YS, ZS, VI, WF, PP, CA} <: AbstractUpwindBiasedAdvectionScheme{2}
    
    "coefficient for ENO reconstruction on x-faces" 
    coeff_xᶠᵃᵃ::XT
    "coefficient for ENO reconstruction on x-centers"
    coeff_xᶜᵃᵃ::XT
    "coefficient for ENO reconstruction on y-faces"
    coeff_yᵃᶠᵃ::YT
    "coefficient for ENO reconstruction on y-centers"
    coeff_yᵃᶜᵃ::YT
    "coefficient for ENO reconstruction on z-faces"
    coeff_zᵃᵃᶠ::ZT
    "coefficient for ENO reconstruction on z-centers"
    coeff_zᵃᵃᶜ::ZT
    
    "coefficient for WENO smoothness indicators on x-faces"
    smooth_xᶠᵃᵃ::XS
    "coefficient for WENO smoothness indicators on x-centers"
    smooth_xᶜᵃᵃ::XS
    "coefficient for WENO smoothness indicators on y-faces"
    smooth_yᵃᶠᵃ::YS
    "coefficient for WENO smoothness indicators on y-centers"
    smooth_yᵃᶜᵃ::YS
    "coefficient for WENO smoothness indicators on z-faces"
    smooth_zᵃᵃᶠ::ZS
    "coefficient for WENO smoothness indicators on z-centers"
    smooth_zᵃᵃᶜ::ZS

    "bounds for maximum-principle-satisfying WENO scheme"
    bounds :: PP

    "advection scheme used near boundaries"
    child_advection :: CA

    function WENO3{FT, VI, WF}(coeff_xᶠᵃᵃ::XT, coeff_xᶜᵃᵃ::XT,
                               coeff_yᵃᶠᵃ::YT, coeff_yᵃᶜᵃ::YT, 
                               coeff_zᵃᵃᶠ::ZT, coeff_zᵃᵃᶜ::ZT,
                               smooth_xᶠᵃᵃ::XS, smooth_xᶜᵃᵃ::XS, 
                               smooth_yᵃᶠᵃ::YS, smooth_yᵃᶜᵃ::YS, 
                               smooth_zᵃᵃᶠ::ZS, smooth_zᵃᵃᶜ::ZS, 
                               bounds::PP, child_advection::CA) where {FT, XT, YT, ZT, XS, YS, ZS, VI, WF, PP, CA}

            return new{FT, XT, YT, ZT, XS, YS, ZS, VI, WF, PP, CA}(coeff_xᶠᵃᵃ, coeff_xᶜᵃᵃ, coeff_yᵃᶠᵃ, coeff_yᵃᶜᵃ, coeff_zᵃᵃᶠ, coeff_zᵃᵃᶜ,
                                                                   smooth_xᶠᵃᵃ, smooth_xᶜᵃᵃ, smooth_yᵃᶠᵃ, smooth_yᵃᶜᵃ, smooth_zᵃᵃᶠ, smooth_zᵃᵃᶜ, 
                                                                   bounds, child_advection)
    end
end

WENO3(grid, FT::DataType=Float64; kwargs...) = WENO3(FT; grid = grid, kwargs...)

function WENO3(FT::DataType = Float64; 
               grid = nothing, 
               stretched_smoothness = false, 
               zweno = true, 
               vector_invariant = nothing,
               bounds = nothing)
    
    if !(grid isa Nothing) 
        FT = eltype(grid)
    end

    weno_coefficients = compute_stretched_weno_coefficients(grid, stretched_smoothness, FT; order = 2)

    VI = typeof(vector_invariant)

    child_advection = UpwindBiasedFirstOrder()

    return WENO3{FT, VI, zweno}(weno_coefficients..., bounds, child_advection)
end

# Flavours of WENO
const ZWENO3        = WENO3{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, true}
const PositiveWENO3 = WENO3{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Tuple}

const WENOVectorInvariantVel3{FT, XT, YT, ZT, XS, YS, ZS, VI, WF, PP}  = 
      WENO3{FT, XT, YT, ZT, XS, YS, ZS, VI, WF, PP} where {FT, XT, YT, ZT, XS, YS, ZS, VI<:VelocityStencil, WF, PP}
const WENOVectorInvariantVort3{FT, XT, YT, ZT, XS, YS, ZS, VI, WF, PP} = 
      WENO3{FT, XT, YT, ZT, XS, YS, ZS, VI, WF, PP} where {FT, XT, YT, ZT, XS, YS, ZS, VI<:VorticityStencil, WF, PP}

const WENOVectorInvariant3 = WENO3{FT, XT, YT, ZT, XS, YS, ZS, VI, WF, PP} where {FT, XT, YT, ZT, XS, YS, ZS, VI<:SmoothnessStencil, WF, PP}

function Base.show(io::IO, a::WENO3{FT, RX, RY, RZ}) where {FT, RX, RY, RZ}
    print(io, "WENO3 advection scheme with: \n",
              "    ├── X $(RX == Nothing ? "regular" : "stretched") \n",
              "    ├── Y $(RY == Nothing ? "regular" : "stretched") \n",
              "    └── Z $(RZ == Nothing ? "regular" : "stretched")" )
end

Adapt.adapt_structure(to, scheme::WENO3{FT, XT, YT, ZT, XS, YS, ZS, VI, WF, PP}) where {FT, XT, YT, ZT, XS, YS, ZS, VI, WF, PP} =
     WENO3{FT, VI, WF}(Adapt.adapt(to, scheme.coeff_xᶠᵃᵃ), Adapt.adapt(to, scheme.coeff_xᶜᵃᵃ),
                       Adapt.adapt(to, scheme.coeff_yᵃᶠᵃ), Adapt.adapt(to, scheme.coeff_yᵃᶜᵃ),
                       Adapt.adapt(to, scheme.coeff_zᵃᵃᶠ), Adapt.adapt(to, scheme.coeff_zᵃᵃᶜ),
                       Adapt.adapt(to, scheme.smooth_xᶠᵃᵃ), Adapt.adapt(to, scheme.smooth_xᶜᵃᵃ),
                       Adapt.adapt(to, scheme.smooth_yᵃᶠᵃ), Adapt.adapt(to, scheme.smooth_yᵃᶜᵃ),
                       Adapt.adapt(to, scheme.smooth_zᵃᵃᶠ), Adapt.adapt(to, scheme.smooth_zᵃᵃᶜ),
                       Adapt.adapt(to, scheme.bounds),
                       Adapt.adapt(to, scheme.child_advection))

@inline boundary_buffer(::WENO3) = 1

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::WENO3, c) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, centered_second_order, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::WENO3, c) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, centered_second_order, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::WENO3, c) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, centered_second_order, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::WENO3, u) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, centered_second_order, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::WENO3, v) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, centered_second_order, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::WENO3, w) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, centered_second_order, w)

# Unroll the functions to pass the coordinates in case of a stretched grid
@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO3, ψ, args...)  = weno3_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, i, Face, args...)
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO3, ψ, args...)  = weno3_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, j, Face, args...)
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO3, ψ, args...)  = weno3_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, k, Face, args...)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO3, ψ, args...) = weno3_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, i, Face, args...)
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO3, ψ, args...) = weno3_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, j, Face, args...)
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO3, ψ, args...) = weno3_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, k, Face, args...)

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO3, ψ, args...)  = weno3_left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, i, Center, args...)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO3, ψ, args...)  = weno3_left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, j, Center, args...)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO3, ψ, args...)  = weno3_left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, k, Center, args...)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO3, ψ, args...) = weno3_right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, i, Center, args...)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO3, ψ, args...) = weno3_right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, j, Center, args...)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO3, ψ, args...) = weno3_right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, k, Center, args...)

# Stencil to calculate the stretched WENO weights and smoothness indicators
@inline left_stencil_x_3(i, j, k, ψ, args...) = @inbounds ( (ψ[i-2, j, k], ψ[i-1, j, k]), (ψ[i-1, j, k], ψ[i, j, k]) )
@inline left_stencil_y_3(i, j, k, ψ, args...) = @inbounds ( (ψ[i, j-2, k], ψ[i, j-1, k]), (ψ[i, j-1, k], ψ[i, j, k]) )
@inline left_stencil_z_3(i, j, k, ψ, args...) = @inbounds ( (ψ[i, j, k-2], ψ[i, j, k-1]), (ψ[i, j, k-1], ψ[i, j, k]) )

@inline right_stencil_x_3(i, j, k, ψ, args...) = @inbounds ( (ψ[i-1, j, k], ψ[i, j, k]), (ψ[i, j, k], ψ[i+1, j, k]) )
@inline right_stencil_y_3(i, j, k, ψ, args...) = @inbounds ( (ψ[i, j-1, k], ψ[i, j, k]), (ψ[i, j, k], ψ[i, j+1, k]) )
@inline right_stencil_z_3(i, j, k, ψ, args...) = @inbounds ( (ψ[i, j, k-1], ψ[i, j, k]), (ψ[i, j, k], ψ[i, j, k+1]) )

@inline left_stencil_x_3(i, j, k, ψ::Function, args...) = @inbounds ( (ψ(i-2, j, k, args...), ψ(i-1, j, k, args...)), (ψ(i-1, j, k, args...), ψ(i, j, k, args...)) )
@inline left_stencil_y_3(i, j, k, ψ::Function, args...) = @inbounds ( (ψ(i, j-2, k, args...), ψ(i, j-1, k, args...)), (ψ(i, j-1, k, args...), ψ(i, j, k, args...)) )
@inline left_stencil_z_3(i, j, k, ψ::Function, args...) = @inbounds ( (ψ(i, j, k-2, args...), ψ(i, j, k-1, args...)), (ψ(i, j, k-1, args...), ψ(i, j, k, args...)) )

@inline right_stencil_x_3(i, j, k, ψ::Function, args...) = @inbounds ( (ψ(i-1, j, k, args...), ψ(i, j, k, args...)), (ψ(i, j, k, args...), ψ(i+1, j, k, args...)) )
@inline right_stencil_y_3(i, j, k, ψ::Function, args...) = @inbounds ( (ψ(i, j-1, k, args...), ψ(i, j, k, args...)), (ψ(i, j, k, args...), ψ(i, j+1, k, args...)) )
@inline right_stencil_z_3(i, j, k, ψ::Function, args...) = @inbounds ( (ψ(i, j, k-1, args...), ψ(i, j, k, args...)), (ψ(i, j, k, args...), ψ(i, j, k+1, args...)) )

# Stencil for vector invariant calculation of smoothness indicators in the horizontal direction

# Parallel to the interpolation direction! (same as left/right stencil)
@inline tangential_left_stencil_u_3(i, j, k, ::Val{1}, u)  = @inbounds left_stencil_x_3(i, j, k, ℑyᵃᶠᵃ, u)
@inline tangential_left_stencil_u_3(i, j, k, ::Val{2}, u)  = @inbounds left_stencil_y_3(i, j, k, ℑyᵃᶠᵃ, u)
@inline tangential_left_stencil_v_3(i, j, k, ::Val{1}, v)  = @inbounds left_stencil_x_3(i, j, k, ℑxᶠᵃᵃ, v)
@inline tangential_left_stencil_v_3(i, j, k, ::Val{2}, v)  = @inbounds left_stencil_y_3(i, j, k, ℑxᶠᵃᵃ, v)

@inline tangential_right_stencil_u_3(i, j, k, ::Val{1}, u)  = @inbounds right_stencil_x_3(i, j, k, ℑyᵃᶠᵃ, u)
@inline tangential_right_stencil_u_3(i, j, k, ::Val{2}, u)  = @inbounds right_stencil_y_3(i, j, k, ℑyᵃᶠᵃ, u)
@inline tangential_right_stencil_v_3(i, j, k, ::Val{1}, v)  = @inbounds right_stencil_x_3(i, j, k, ℑxᶠᵃᵃ, v)
@inline tangential_right_stencil_v_3(i, j, k, ::Val{2}, v)  = @inbounds right_stencil_y_3(i, j, k, ℑxᶠᵃᵃ, v)

#####
##### Jiang & Shu (1996) WENO smoothness indicators. See also Equation 2.63 in Shu (1998)
#####

@inline left_biased_β₀(FT, ψ, ::Type{Nothing}, scheme::WENO3, args...) = @inbounds (ψ[2] - ψ[1])^two_32
@inline left_biased_β₁(FT, ψ, ::Type{Nothing}, scheme::WENO3, args...) = @inbounds (ψ[2] - ψ[1])^two_32

@inline right_biased_β₀(FT, ψ, ::Type{Nothing}, scheme::WENO3, args...) = @inbounds (ψ[2] - ψ[1])^two_32
@inline right_biased_β₁(FT, ψ, ::Type{Nothing}, scheme::WENO3, args...) = @inbounds (ψ[2] - ψ[1])^two_32

#####
##### VectorInvariant reconstruction (based on JS or Z) (z-direction Val{3} is different from x- and y-directions)
##### JS-WENO-5 reconstruction
#####

for (side, coeffs) in zip([:left, :right], ([:C2₀, :C2₁], [:C2₁, :C2₀]))
    biased_weno3_weights = Symbol(side, :_biased_weno3_weights)
    
    biased_β₀ = Symbol(side, :_biased_β₀)
    biased_β₁ = Symbol(side, :_biased_β₁)

    tangential_stencil_u = Symbol(:tangential_, side, :_stencil_u_3)
    tangential_stencil_v = Symbol(:tangential_, side, :_stencil_v_3)
    
    @eval begin
        @inline function $biased_weno3_weights(FT, ψₜ, T, scheme, dir, idx, loc, args...)
            ψ₁, ψ₀ = ψₜ 
            β₀ = $biased_β₀(FT, ψ₀, T, scheme, dir, idx, loc)
            β₁ = $biased_β₁(FT, ψ₁, T, scheme, dir, idx, loc)

            if scheme isa ZWENO3
                τ₅ = abs(β₁ - β₀)
                α₀ = FT($(coeffs[1])) * (1 + (τ₅ / (β₀ + FT(ε)))^ƞ) 
                α₁ = FT($(coeffs[2])) * (1 + (τ₅ / (β₁ + FT(ε)))^ƞ) 
            else
                α₀ = FT($(coeffs[1])) / (β₀ + FT(ε))^ƞ
                α₁ = FT($(coeffs[2])) / (β₁ + FT(ε))^ƞ
            end

            Σα = α₀ + α₁
            w₀ = α₀ / Σα
            w₁ = α₁ / Σα
        
            return w₀, w₁
        end

        @inline function $biased_weno3_weights(FT, ijk, T, scheme, dir, idx, loc, ::Type{VelocityStencil}, u, v)
            i, j, k = ijk
            
            u₁, u₀ = $tangential_stencil_u(i, j, k, dir, u)
            v₁, v₀ = $tangential_stencil_v(i, j, k, dir, v)
    
            βu₀ = $biased_β₀(FT, u₀, T, scheme, Val(2), idx, loc)
            βu₁ = $biased_β₁(FT, u₁, T, scheme, Val(2), idx, loc)
        
            βv₀ = $biased_β₀(FT, v₀, T, scheme, Val(1), idx, loc)
            βv₁ = $biased_β₁(FT, v₁, T, scheme, Val(1), idx, loc)
                   
            β₀ = 0.5*(βu₀ + βv₀)  
            β₁ = 0.5*(βu₁ + βv₁)     
        
            if scheme isa ZWENO3
                τ₅ = abs(β₁ - β₀)
                α₀ = FT($(coeffs[1])) * (1 + (τ₅ / (β₀ + FT(ε)))^ƞ) 
                α₁ = FT($(coeffs[2])) * (1 + (τ₅ / (β₁ + FT(ε)))^ƞ) 
            else
                α₀ = FT($(coeffs[1])) / (β₀ + FT(ε))^ƞ
                α₁ = FT($(coeffs[2])) / (β₁ + FT(ε))^ƞ
            end
                
            Σα = α₀ + α₁
            w₀ = α₀ / Σα
            w₁ = α₁ / Σα
        
            return w₀, w₁
        end
    end
end

#####
##### Biased interpolation functions
#####

for (interp, dir, val, cT, cS) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [:x, :y, :z], [1, 2, 3], [:XT, :YT, :ZT], [:XS, :YS, :ZS]) 
    for side in (:left, :right)
        interpolate_func = Symbol(:weno3_, side, :_biased_interpolate_, interp)
        stencil       = Symbol(side, :_stencil_, dir, :_3)
        weno3_weights = Symbol(side, :_biased_weno3_weights)
        biased_p₀ = Symbol(side, :_biased_p₀)
        biased_p₁ = Symbol(side, :_biased_p₁)

        @eval begin
            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENO3{FT, XT, YT, ZT, XS, YS, ZS}, 
                                               ψ, idx, loc, args...) where {FT, XT, YT, ZT, XS, YS, ZS}
                
                ψ₁, ψ₀ = ψₜ = $stencil(i, j, k, ψ, grid, args...)
                w₀, w₁ = $weno3_weights(FT, pass_stencil(ψₜ, i, j, k, Nothing), $cS, scheme, Val($val), idx, loc, Nothing, args...)
                return w₀ * $biased_p₀(scheme, ψ₀, $cT, Val($val), idx, loc) + 
                       w₁ * $biased_p₁(scheme, ψ₁, $cT, Val($val), idx, loc) 
            end

            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENOVectorInvariant3{FT, XT, YT, ZT, XS, YS, ZS}, 
                                               ψ, idx, loc, VI, args...) where {FT, XT, YT, ZT, XS, YS, ZS}

                ψ₁, ψ₀ = ψₜ = $stencil(i, j, k, ψ, grid, args...)
                w₀, w₁ = $weno3_weights(FT, pass_stencil(ψₜ, i, j, k, VI), $cS, scheme, Val($val), idx, loc, VI, args...)
                return w₀ * $biased_p₀(scheme, ψ₀, $cT, Val($val), idx, loc) + 
                       w₁ * $biased_p₁(scheme, ψ₁, $cT, Val($val), idx, loc) 
            end
        end
    end
end

#####
##### Coefficients for stretched (and uniform) ENO schemes (see Shu NASA/CR-97-206253, ICASE Report No. 97-65)
#####

@inline coeff_left_p₀(scheme::WENO3{FT}, ::Type{Nothing}, args...) where FT = (  FT(1/2), FT(1/2))
@inline coeff_left_p₁(scheme::WENO3{FT}, ::Type{Nothing}, args...) where FT = (- FT(1/2), FT(3/2))

@inline coeff_right_p₀(scheme::WENO3, ::Type{Nothing}, args...) = reverse(coeff_left_p₁(scheme, Nothing, args...)) 
@inline coeff_right_p₁(scheme::WENO3, ::Type{Nothing}, args...) = reverse(coeff_left_p₀(scheme, Nothing, args...)) 
