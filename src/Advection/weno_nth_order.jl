#####
##### Weighted Essentially Non-Oscillatory (WENO) fifth-order advection scheme
#####

const two_32 = Int32(2)

const ƞ = Int32(2) # WENO exponent
const ε = 1e-6

abstract type SmoothnessStencil end

struct VorticityStencil <:SmoothnessStencil end
struct VelocityStencil <:SmoothnessStencil end

struct WENO{N, FT, XT, YT, ZT, VI, WF, PP, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N}
    
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

    "bounds for maximum-principle-satisfying WENO scheme"
    bounds :: PP

    "advection scheme used near boundaries"
    boundary_scheme :: CA
    symmetric_scheme :: SI

    function WENO{N, FT, VI, WF}(coeff_xᶠᵃᵃ::XT, coeff_xᶜᵃᵃ::XT,
                                 coeff_yᵃᶠᵃ::YT, coeff_yᵃᶜᵃ::YT, 
                                 coeff_zᵃᵃᶠ::ZT, coeff_zᵃᵃᶜ::ZT,
                                 bounds::PP, boundary_scheme::CA,
                                 symmetric_scheme :: SI) where {N, FT, XT, YT, ZT, VI, WF, PP, CA, SI}

            return new{N, FT, XT, YT, ZT, VI, WF, PP, CA, SI}(coeff_xᶠᵃᵃ, coeff_xᶜᵃᵃ, 
                                                              coeff_yᵃᶠᵃ, coeff_yᵃᶜᵃ, 
                                                              coeff_zᵃᵃᶠ, coeff_zᵃᵃᶜ,
                                                              bounds, boundary_scheme, symmetric_scheme)
    end
end

WENO(grid, FT::DataType=Float64; kwargs...) = WENO(FT; grid = grid, kwargs...)

function WENO(FT::DataType = Float64; 
               order = 5,
               grid = nothing, 
               zweno = true, 
               vector_invariant = nothing,
               bounds = nothing)
    
    if !(grid isa Nothing) 
        FT = eltype(grid)
    end

    if order < 3
        return UpwindBiasedFirstOrder()
    else
        VI = typeof(vector_invariant)
        N  = Int((order + 1) /2)

        weno_coefficients = compute_stretched_weno_coefficients(grid, false, FT; order = N)
        boundary_scheme = WENO(FT; grid, order = order - 2, zweno, vector_invariant, bounds)
        if N > 2
            symmetric_scheme = CenteredFourthOrder()
        else
            symmetric_scheme = CenteredSecondOrder()
        end
    end

    return WENO{N, FT, VI, zweno}(weno_coefficients[1:6]..., bounds, boundary_scheme, symmetric_scheme)
end

# Flavours of WENO
const ZWENO        = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, true}
const PositiveWENO = WENO{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Tuple}

const WENOVectorInvariantVel{N, FT, XT, YT, ZT, VI, WF, PP}  = 
      WENO{N, FT, XT, YT, ZT, VI, WF, PP} where {N, FT, XT, YT, ZT, VI<:VelocityStencil, WF, PP}
const WENOVectorInvariantVort{N, FT, XT, YT, ZT, VI, WF, PP} = 
      WENO{N, FT, XT, YT, ZT, VI, WF, PP} where {N, FT, XT, YT, ZT, VI<:VorticityStencil, WF, PP}

const WENOVectorInvariant{N, FT, XT, YT, ZT, VI, WF, PP} = 
      WENO{N, FT, XT, YT, ZT, VI, WF, PP} where {N, FT, XT, YT, ZT, VI<:SmoothnessStencil, WF, PP}

function Base.show(io::IO, a::WENO{N, FT, RX, RY, RZ}) where {N, FT, RX, RY, RZ}
    print(io, "WENO advection scheme order $(N*2 -1): \n",
              "    ├── X $(RX == Nothing ? "regular" : "stretched") \n",
              "    ├── Y $(RY == Nothing ? "regular" : "stretched") \n",
              "    └── Z $(RZ == Nothing ? "regular" : "stretched")" )
end

Adapt.adapt_structure(to, scheme::WENO{N, FT, XT, YT, ZT, VI, WF, PP}) where {N, FT, XT, YT, ZT, VI, WF, PP} =
     WENO{N, FT, VI, WF}(Adapt.adapt(to, scheme.coeff_xᶠᵃᵃ), Adapt.adapt(to, scheme.coeff_xᶜᵃᵃ),
                         Adapt.adapt(to, scheme.coeff_yᵃᶠᵃ), Adapt.adapt(to, scheme.coeff_yᵃᶜᵃ),
                         Adapt.adapt(to, scheme.coeff_zᵃᵃᶠ), Adapt.adapt(to, scheme.coeff_zᵃᵃᶜ),
                         Adapt.adapt(to, scheme.bounds),
                         Adapt.adapt(to, scheme.boundary_scheme),
                         Adapt.adapt(to, scheme.symmetric_scheme))

@inline boundary_buffer(::WENO{N}) where N = N

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO, c) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.symmetric_scheme, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO, c) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.symmetric_scheme, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO, c) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.symmetric_scheme, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO, u) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.symmetric_scheme, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO, v) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.symmetric_scheme, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO, w) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme.symmetric_scheme, w)

# Unroll the functions to pass the coordinates in case of a stretched grid
@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO, ψ, args...)  = weno_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, i, Face, args...)
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO, ψ, args...)  = weno_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, j, Face, args...)
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO, ψ, args...)  = weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, k, Face, args...)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO, ψ, args...) = weno_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, ψ, i, Face, args...)
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO, ψ, args...) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, ψ, j, Face, args...)
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO, ψ, args...) = weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, ψ, k, Face, args...)

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO, ψ, args...)  = weno_left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, i, Center, args...)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO, ψ, args...)  = weno_left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, j, Center, args...)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO, ψ, args...)  = weno_left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, k, Center, args...)

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::WENO, ψ, args...) = weno_right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, ψ, i, Center, args...)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::WENO, ψ, args...) = weno_right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, ψ, j, Center, args...)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::WENO, ψ, args...) = weno_right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, ψ, k, Center, args...)

function calc_stencil(buffer, shift, dir; func = false) 
    N = buffer * 2
    if shift != :none
        N -=1
    end
    stencil_full = Vector(undef, buffer)
    rng = 1:N
    if shift == :right
        rng = rng .+ 1
    end
    for stencil in 1:buffer
        stencil_point = Vector(undef, buffer)
        rngstencil = rng[stencil:stencil+buffer-1]
        for (idx, n) in enumerate(rngstencil)
            c = n - buffer - 1
            if func 
                stencil_point[idx] =  dir == :x ? 
                                    :(ψ(i + $c, j, k, args...)) :
                                    dir == :y ?
                                    :(ψ(i, j + $c, k, args...)) :
                                    :(ψ(i, j, k + $c, args...))
            else    
                stencil_point[idx] =  dir == :x ? 
                                    :(ψ[i + $c, j, k]) :
                                    dir == :y ?
                                    :(ψ[i, j + $c, k]) :
                                    :(ψ[i, j, k + $c])
            end                
        end
        stencil_full[stencil] = :($(stencil_point...), )
    end
    return stencil_full
end

for side in (:left, :right), dir in (:x, :y, :z)
    stencil = Symbol(side, :_stencil_, dir)

    for buffer in [2, 3, 4, 5, 6]
        @eval begin
            $stencil(i, j, k, scheme::WENO{$buffer}, ψ, args...)           = @inbounds ($(calc_stencil(buffer, side, dir)...),)
            $stencil(i, j, k, scheme::WENO{$buffer}, ψ::Function, args...) = @inbounds ($(calc_stencil(buffer, side, dir; func = true)...),)
        end
    end
end

# Stencil for vector invariant calculation of smoothness indicators in the horizontal direction
# Parallel to the interpolation direction! (same as left/right stencil)
@inline tangential_left_stencil_u(i, j, k, scheme::WENO, ::Val{1}, u)  = @inbounds left_stencil_x(i, j, k, scheme, ℑyᵃᶠᵃ, u)
@inline tangential_left_stencil_u(i, j, k, scheme::WENO, ::Val{2}, u)  = @inbounds left_stencil_y(i, j, k, scheme, ℑyᵃᶠᵃ, u)
@inline tangential_left_stencil_v(i, j, k, scheme::WENO, ::Val{1}, v)  = @inbounds left_stencil_x(i, j, k, scheme, ℑxᶠᵃᵃ, v)
@inline tangential_left_stencil_v(i, j, k, scheme::WENO, ::Val{2}, v)  = @inbounds left_stencil_y(i, j, k, scheme, ℑxᶠᵃᵃ, v)

@inline tangential_right_stencil_u(i, j, k, scheme::WENO, ::Val{1}, u)  = @inbounds right_stencil_x(i, j, k, scheme, ℑyᵃᶠᵃ, u)
@inline tangential_right_stencil_u(i, j, k, scheme::WENO, ::Val{2}, u)  = @inbounds right_stencil_y(i, j, k, scheme, ℑyᵃᶠᵃ, u)
@inline tangential_right_stencil_v(i, j, k, scheme::WENO, ::Val{1}, v)  = @inbounds right_stencil_x(i, j, k, scheme, ℑxᶠᵃᵃ, v)
@inline tangential_right_stencil_v(i, j, k, scheme::WENO, ::Val{2}, v)  = @inbounds right_stencil_y(i, j, k, scheme, ℑxᶠᵃᵃ, v)
