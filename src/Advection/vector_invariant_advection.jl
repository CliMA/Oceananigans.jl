using Oceananigans.Operators
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, Γᶠᶠᶜ

struct EnergyConservingScheme{FT}    <: AbstractAdvectionScheme{1, FT} end
struct EnstrophyConservingScheme{FT} <: AbstractAdvectionScheme{1, FT} end

EnergyConservingScheme(FT::DataType = Float64)    = EnergyConservingScheme{FT}()
EnstrophyConservingScheme(FT::DataType = Float64) = EnstrophyConservingScheme{FT}()

struct VectorInvariant{N, FT, Z, ZS, V, M} <: AbstractAdvectionScheme{N, FT}
    "reconstruction scheme for vorticity flux"
    vorticity_scheme   :: Z
    "reconstruction scheme for divergence flux"
    vorticity_stencil  :: ZS
    "stencil used for assessing divergence smoothness"
    vertical_scheme    :: V
    
    function VectorInvariant{N, FT, M}(vorticity_scheme::Z, vorticity_stencil::ZS, vertical_scheme::V) where {N, FT, Z, ZS, V, M}
        return new{N, FT, Z, ZS, V, M}(vorticity_scheme, vorticity_stencil, vertical_scheme)
    end
end

"""
    VectorInvariant(; vorticity_scheme::AbstractAdvectionScheme{N, FT} = EnstrophyConservingScheme(), 
                      vorticity_stencil  = VelocityStencil(),
                      vertical_scheme    = EnergyConservingScheme()) where {N, FT}
               
Construct a vector invariant momentum advection scheme of order `N * 2 - 1`.

Keyword arguments
=================

- `vorticity_scheme`: Scheme used for `Center` reconstruction of vorticity, options are upwind advection schemes
                      - `UpwindBiased` and `WENO` - in addition to an `EnergyConservingScheme` and an `EnstrophyConservingScheme`
                      (defaults to `EnstrophyConservingScheme`)
- `vorticity_stencil`: Stencil used for smoothness indicators in case of a `WENO` upwind reconstruction. Choices are between `VelocityStencil`
                       which uses the horizontal velocity field to diagnose smoothness and `DefaultStencil` which uses the variable
                       being transported (defaults to `VelocityStencil`)
- `vertical_scheme`: Scheme used for vertical advection of horizontal momentum. It has to be consistent with the choice of 
                     `divergence_stencil`. If the latter is a `Nothing`, only `EnergyConservingScheme` is available (this keyword
                     argument has no effect). In case `divergence_scheme` is an `AbstractUpwindBiasedAdvectionScheme`, 
                     `vertical_scheme` describes a flux form reconstruction of vertical momentum advection, and any 
                     advection scheme can be used - `Centered`, `UpwindBiased` and `WENO` (defaults to `EnergyConservingScheme`)
- `multi_dimensional_stencil` : use a horizontal two dimensional stencil for the reconstruction of vorticity and divergence.
                                The tangential (not upwinded) direction is treated with a 5th order centered WENO reconstruction

Examples
========
```jldoctest
julia> using Oceananigans

julia> VectorInvariant()
Vector Invariant reconstruction, maximum order 1 
 Vorticity flux scheme: 
    └── EnstrophyConservingScheme{Float64} 
 Vertical advection scheme: 
    └── EnergyConservingScheme{Float64}

```
```jldoctest
julia> using Oceananigans

julia> VectorInvariant(vorticity_scheme = WENO(), vertical_scheme = WENO(order = 3))
Vector Invariant reconstruction, maximum order 5 
 Vorticity flux scheme: 
    └── WENO reconstruction order 5 with smoothness stencil Oceananigans.Advection.VelocityStencil()
 Vertical advection scheme: 
    └── WENO reconstruction order 3
```
"""
function VectorInvariant(; vorticity_scheme::AbstractAdvectionScheme{N, FT} = EnstrophyConservingScheme(), 
                           vorticity_stencil  = VelocityStencil(),
                           vertical_scheme    = EnergyConservingScheme(),
                           multi_dimensional_stencil = false) where {N, FT}
        
    return VectorInvariant{N, FT, multi_dimensional_stencil}(vorticity_scheme, vorticity_stencil, vertical_scheme)
end

Base.summary(a::VectorInvariant{N}) where N = string("Vector Invariant reconstruction, maximum order ", N*2-1)

Base.show(io::IO, a::VectorInvariant{N, FT}) where {N, FT} =
    print(io, summary(a), " \n",
              " Vorticity flux scheme: ", "\n",
              "    └── $(summary(a.vorticity_scheme)) $(a.vorticity_scheme isa WENO ? "with smoothness stencil $(a.vorticity_stencil)" : "")\n",
              " Vertical advection scheme: ", "\n",
              "    └── $(summary(a.vertical_scheme))")

# Since vorticity itself requires one halo, if we use an upwinding scheme (N > 1) we require one additional
# halo for vector invariant advection
required_halo_size(scheme::VectorInvariant{N}) where N = N == 1 ? N : N + 1

Adapt.adapt_structure(to, scheme::VectorInvariant{N, FT, Z, ZS, V, M}) where {N, FT, Z, ZS, V, M} =
        VectorInvariant{N, FT, M}(Adapt.adapt(to, scheme.vorticity_scheme), 
                                  Adapt.adapt(to, scheme.vorticity_stencil), 
                                  Adapt.adapt(to, scheme.vertical_scheme))

@inline vertical_scheme(scheme::VectorInvariant) = string(nameof(typeof(scheme.vertical_scheme)))

const VectorInvariantEnergyConserving    = VectorInvariant{<:Any, <:Any, <:EnergyConservingScheme}
const VectorInvariantEnstrophyConserving = VectorInvariant{<:Any, <:Any, <:EnstrophyConservingScheme}

const VectorInvariantVerticallyEnergyConserving  = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:EnergyConservingScheme}

const UpwindVorticityVectorInvariant        = VectorInvariant{<:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme}
const MultiDimensionalUpwindVectorInvariant = VectorInvariant{<:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:Any, <:AbstractUpwindBiasedAdvectionScheme, true}

@inline U_dot_∇u(i, j, k, grid, scheme::VectorInvariant, U) = (
    + horizontal_advection_U(i, j, k, grid, scheme, U.u, U.v)
    + vertical_advection_U(i, j, k, grid, scheme, U.w, U.u, U.v)
    + bernoulli_head_U(i, j, k, grid, scheme, U.u, U.v))
    
@inline U_dot_∇v(i, j, k, grid, scheme::VectorInvariant, U) = (
    + horizontal_advection_V(i, j, k, grid, scheme, U.u, U.v)
    + vertical_advection_V(i, j, k, grid, scheme, U.w, U.u, U.v)
    + bernoulli_head_V(i, j, k, grid, scheme, U.u, U.v))

#####
#####  Vertical advection + Kinetic Energy gradient. 3 Formulations:
#####  1. Energy conserving
#####  2. Dimension-By-Dimension Divergence + KE upwinding   
#####  3. Multi-Dimensional Divergence + KE upwinding     
#####

#####
##### Conservative vertical advection + Kinetic Energy gradient (1)
##### Follows https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#vector-invariant-momentum-equations
#####

@inline ϕ²(i, j, k, grid, ϕ)       = @inbounds ϕ[i, j, k]^2
@inline Khᶜᶜᶜ(i, j, k, grid, u, v) = (ℑxᶜᵃᵃ(i, j, k, grid, ϕ², u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², v)) / 2

@inline bernoulli_head_U(i, j, k, grid, ::VectorInvariantVerticallyEnergyConserving, u, v) = ∂xᶠᶜᶜ(i, j, k, grid, Khᶜᶜᶜ, u, v)
@inline bernoulli_head_V(i, j, k, grid, ::VectorInvariantVerticallyEnergyConserving, u, v) = ∂yᶜᶠᶜ(i, j, k, grid, Khᶜᶜᶜ, u, v)
    
@inbounds ζ₂wᶠᶜᶠ(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶠᶜᶠ(i, j, k, grid, u) 
@inbounds ζ₁wᶜᶠᶠ(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶜᶠᶠ(i, j, k, grid, v) 

@inline vertical_advection_U(i, j, k, grid, ::VectorInvariantVerticallyEnergyConserving, w, u, v) =  ℑzᵃᵃᶜ(i, j, k, grid, ζ₂wᶠᶜᶠ, u, w) / Azᶠᶜᶜ(i, j, k, grid)
@inline vertical_advection_V(i, j, k, grid, ::VectorInvariantVerticallyEnergyConserving, w, u, v) =  ℑzᵃᵃᶜ(i, j, k, grid, ζ₁wᶜᶠᶠ, v, w) / Azᶜᶠᶜ(i, j, k, grid)

#####
##### Upwinding vertical advection + Kinetic Energy (2. and 3.)
#####

@inline function upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariant, u, v)
    @inbounds û = u[i, j, k]
    δvˢ =    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.vertical_scheme, δyᵃᶜᵃ, Ay_qᶜᶠᶜ, v) 
    δuᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.vertical_scheme, δxᶜᵃᵃ, Ax_qᶠᶜᶜ, u) 
    δuᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.vertical_scheme, δxᶜᵃᵃ, Ax_qᶠᶜᶜ, u) 

    return upwind_biased_product(û, δuᴸ, δuᴿ) + û * δvˢ
end

@inline function upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariant, u, v)
    @inbounds v̂ = v[i, j, k]
    δuˢ =    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.vertical_scheme, δxᶜᵃᵃ, Ax_qᶠᶜᶜ, u) 
    δvᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.vertical_scheme, δyᵃᶜᵃ, Ay_qᶜᶠᶜ, v) 
    δvᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.vertical_scheme, δyᵃᶜᵃ, Ay_qᶜᶠᶜ, v) 

    return upwind_biased_product(v̂, δvᴸ, δvᴿ) + v̂ * δuˢ
end

@inline function upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::MultiDimensionalUpwindVectorInvariant, u, v)
    @inbounds û = u[i, j, k]
    δvˢ = _multi_dimensional_reconstruction_y(i, j, k, grid, scheme.vertical_scheme,    _symmetric_interpolate_xᶠᵃᵃ, δyᵃᶜᵃ, Ay_qᶜᶠᶜ, v) 
    δuᴸ = _multi_dimensional_reconstruction_y(i, j, k, grid, scheme.vertical_scheme,  _left_biased_interpolate_xᶠᵃᵃ, δxᶜᵃᵃ, Ax_qᶠᶜᶜ, u) 
    δuᴿ = _multi_dimensional_reconstruction_y(i, j, k, grid, scheme.vertical_scheme, _right_biased_interpolate_xᶠᵃᵃ, δxᶜᵃᵃ, Ax_qᶠᶜᶜ, u) 

    return upwind_biased_product(û, δuᴸ, δuᴿ) + û * δvˢ
end

@inline function upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::MultiDimensionalUpwindVectorInvariant, u, v)
    @inbounds v̂ = v[i, j, k]
    δuˢ = _multi_dimensional_reconstruction_x(i, j, k, grid, scheme.vertical_scheme,    _symmetric_interpolate_yᵃᶠᵃ, δxᶜᵃᵃ, Ax_qᶠᶜᶜ, u) 
    δvᴸ = _multi_dimensional_reconstruction_x(i, j, k, grid, scheme.vertical_scheme,  _left_biased_interpolate_yᵃᶠᵃ, δyᵃᶜᵃ, Ay_qᶜᶠᶜ, v) 
    δvᴿ = _multi_dimensional_reconstruction_x(i, j, k, grid, scheme.vertical_scheme, _right_biased_interpolate_yᵃᶠᵃ, δyᵃᶜᵃ, Ay_qᶜᶠᶜ, v) 

    return upwind_biased_product(v̂, δvᴸ, δvᴿ) + v̂ * δuˢ
end

@inline function vertical_advection_U(i, j, k, grid, scheme::VectorInvariant, w, u, v) 
    
    δt = upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid,       scheme.vertical_scheme, u, v)
    ca = δzᵃᵃᶜ(i, j, k, grid, advective_momentum_flux_Wu, scheme.vertical_scheme, w, u)

    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δt + ca)
end

@inline function vertical_advection_V(i, j, k, grid, scheme::VectorInvariant, w, u, v) 

    δt = upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid,       scheme.vertical_scheme, u, v)
    ca = δzᵃᵃᶜ(i, j, k, grid, advective_momentum_flux_Wv, scheme.vertical_scheme, w, v)

    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δt + ca)
end

@inline half_ϕ²(i, j, k, grid, ϕ) = ϕ[i, j, k]^2 / 2

@inline function bernoulli_head_U(i, j, k, grid, scheme, u, v)

    @inbounds û = u[i, j, k]
    δKvˢ =    _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.vertical_scheme, δxᶠᵃᵃ, half_ϕ², v) 
    δKuᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.vertical_scheme, δxᶜᵃᵃ, half_ϕ², u)
    δKuᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.vertical_scheme, δxᶜᵃᵃ, half_ϕ², u)
    
    ∂Kᴸ = (δKuᴸ + δKvˢ) / Δxᶠᶜᶜ(i, j, k, grid)
    ∂Kᴿ = (δKuᴿ + δKvˢ) / Δxᶠᶜᶜ(i, j, k, grid)

    return ifelse(û > 0, ∂Kᴸ, ∂Kᴿ)
end

@inline function bernoulli_head_V(i, j, k, grid, scheme, u, v)

    @inbounds v̂ = v[i, j, k]
    δKuˢ =    _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.vertical_scheme, δyᵃᶠᵃ, half_ϕ², u)
    δKvᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.vertical_scheme, δyᵃᶜᵃ, half_ϕ², v) 
    δKvᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.vertical_scheme, δyᵃᶜᵃ, half_ϕ², v) 
    
    ∂Kᴸ = (δKvᴸ + δKuˢ) / Δyᶜᶠᶜ(i, j, k, grid) 
    ∂Kᴿ = (δKvᴿ + δKuˢ) / Δyᶜᶠᶜ(i, j, k, grid)

    return ifelse(v̂ > 0, ∂Kᴸ, ∂Kᴿ)
end

@inline function bernoulli_head_U(i, j, k, grid, scheme::MultiDimensionalUpwindVectorInvariant, u, v)

    @inbounds û = u[i, j, k]
    δKvˢ = _multi_dimensional_reconstruction_x(i, j, k, grid, scheme.vertical_scheme,    _symmetric_interpolate_yᵃᶜᵃ, δxᶠᵃᵃ, half_ϕ², v) 
    δKuᴸ = _multi_dimensional_reconstruction_y(i, j, k, grid, scheme.vertical_scheme,  _left_biased_interpolate_xᶠᵃᵃ, δxᶜᵃᵃ, half_ϕ², u)
    δKuᴿ = _multi_dimensional_reconstruction_y(i, j, k, grid, scheme.vertical_scheme, _right_biased_interpolate_xᶠᵃᵃ, δxᶜᵃᵃ, half_ϕ², u)
    
    ∂Kᴸ = (δKuᴸ + δKvˢ) / Δxᶠᶜᶜ(i, j, k, grid)
    ∂Kᴿ = (δKuᴿ + δKvˢ) / Δxᶠᶜᶜ(i, j, k, grid)

    return ifelse(û > 0, ∂Kᴸ, ∂Kᴿ)
end

@inline function bernoulli_head_V(i, j, k, grid, scheme::MultiDimensionalUpwindVectorInvariant, u, v)

    @inbounds v̂ = v[i, j, k]
    δKuˢ = _multi_dimensional_reconstruction_y(i, j, k, grid, scheme.vertical_scheme,    _symmetric_interpolate_xᶜᵃᵃ, δyᵃᶠᵃ, half_ϕ², u)
    δKvᴸ = _multi_dimensional_reconstruction_x(i, j, k, grid, scheme.vertical_scheme,  _left_biased_interpolate_yᵃᶠᵃ, δyᵃᶜᵃ, half_ϕ², v) 
    δKvᴿ = _multi_dimensional_reconstruction_x(i, j, k, grid, scheme.vertical_scheme, _right_biased_interpolate_yᵃᶠᵃ, δyᵃᶜᵃ, half_ϕ², v) 
    
    ∂Kᴸ = (δKvᴸ + δKuˢ) / Δyᶜᶠᶜ(i, j, k, grid) 
    ∂Kᴿ = (δKvᴿ + δKuˢ) / Δyᶜᶠᶜ(i, j, k, grid)

    return ifelse(v̂ > 0, ∂Kᴸ, ∂Kᴿ)
end

#####
##### Horizontal advection 4 formulations:
#####  1. Energy conservative         
#####  2. Enstrophy conservative      
#####  3. Dimension-By-Dimension Vorticity upwinding   
#####  4. Multi-Dimensional Vorticity upwinding         
#####

#####
##### Conserving schemes (1. and 2.)
##### Follows https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#vector-invariant-momentum-equations
#####

@inline ζ_ℑx_vᶠᶠᵃ(i, j, k, grid, u, v) = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)
@inline ζ_ℑy_uᶠᶠᵃ(i, j, k, grid, u, v) = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)

@inline horizontal_advection_U(i, j, k, grid, ::VectorInvariantEnergyConserving, u, v) = - ℑyᵃᶜᵃ(i, j, k, grid, ζ_ℑx_vᶠᶠᵃ, u, v) / Δxᶠᶜᶜ(i, j, k, grid)
@inline horizontal_advection_V(i, j, k, grid, ::VectorInvariantEnergyConserving, u, v) = + ℑxᶜᵃᵃ(i, j, k, grid, ζ_ℑy_uᶠᶠᵃ, u, v) / Δyᶜᶠᶜ(i, j, k, grid)

@inline horizontal_advection_U(i, j, k, grid, ::VectorInvariantEnstrophyConserving, u, v) = - ℑyᵃᶜᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 
@inline horizontal_advection_V(i, j, k, grid, ::VectorInvariantEnstrophyConserving, u, v) = + ℑxᶜᵃᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)

#####
##### Upwinding schemes (3. and 4.)
#####

@inline function horizontal_advection_U(i, j, k, grid, scheme::UpwindVorticityVectorInvariant, u, v)
    
    Sζ = scheme.vorticity_stencil

    @inbounds v̂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 
    ζᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v)
    ζᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v)

    return - upwind_biased_product(v̂, ζᴸ, ζᴿ)
end

@inline function horizontal_advection_V(i, j, k, grid, scheme::UpwindVorticityVectorInvariant, u, v) 

    Sζ = scheme.vorticity_stencil

    @inbounds û  =  ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)
    ζᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v)
    ζᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v)

    return + upwind_biased_product(û, ζᴸ, ζᴿ)
end

@inline function horizontal_advection_U(i, j, k, grid, scheme::MultiDimensionalUpwindVectorInvariant, u, v)
    
    Sζ = scheme.vorticity_stencil

    @inbounds v̂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 
    ζᴸ = _multi_dimensional_reconstruction_x(i, j, k, grid, scheme.vorticity_scheme,  _left_biased_interpolate_yᵃᶜᵃ, ζ₃ᶠᶠᶜ, Sζ, u, v)
    ζᴿ = _multi_dimensional_reconstruction_x(i, j, k, grid, scheme.vorticity_scheme, _right_biased_interpolate_yᵃᶜᵃ, ζ₃ᶠᶠᶜ, Sζ, u, v)

    return - upwind_biased_product(v̂, ζᴸ, ζᴿ)
end

@inline function horizontal_advection_V(i, j, k, grid, scheme::MultiDimensionalUpwindVectorInvariant, u, v) 

    Sζ = scheme.vorticity_stencil

    @inbounds û  =  ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)
    ζᴸ = _multi_dimensional_reconstruction_y(i, j, k, grid, scheme.vorticity_scheme,  _left_biased_interpolate_xᶜᵃᵃ, ζ₃ᶠᶠᶜ, Sζ, u, v)
    ζᴿ = _multi_dimensional_reconstruction_y(i, j, k, grid, scheme.vorticity_scheme, _right_biased_interpolate_xᶜᵃᵃ, ζ₃ᶠᶠᶜ, Sζ, u, v)

    return + upwind_biased_product(û, ζᴸ, ζᴿ) 
end

#####
##### Fallback
#####

@inline U_dot_∇u(i, j, k, grid, scheme::AbstractAdvectionScheme, U) = div_𝐯u(i, j, k, grid, scheme, U, U.u)
@inline U_dot_∇v(i, j, k, grid, scheme::AbstractAdvectionScheme, U) = div_𝐯v(i, j, k, grid, scheme, U, U.v)

#####
##### No advection
#####

@inline U_dot_∇u(i, j, k, grid::AbstractGrid{FT}, scheme::Nothing, U) where FT = zero(FT)
@inline U_dot_∇v(i, j, k, grid::AbstractGrid{FT}, scheme::Nothing, U) where FT = zero(FT)

const U{N}  = UpwindBiased{N}
const UX{N} = UpwindBiased{N, <:Any, <:Nothing} 
const UY{N} = UpwindBiased{N, <:Any, <:Any, <:Nothing}
const UZ{N} = UpwindBiased{N, <:Any, <:Any, <:Any, <:Nothing}

# To adapt passing smoothness stencils to upwind biased schemes (not weno) 
for buffer in 1:6
    @eval begin
        @inline inner_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::U{$buffer},  f::Function, idx, loc, VI::AbstractSmoothnessStencil, args...) = inner_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, f, idx, loc, args...)
        @inline inner_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::UX{$buffer}, f::Function, idx, loc, VI::AbstractSmoothnessStencil, args...) = inner_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, f, idx, loc, args...)
        @inline inner_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::U{$buffer},  f::Function, idx, loc, VI::AbstractSmoothnessStencil, args...) = inner_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, f, idx, loc, args...)
        @inline inner_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::UY{$buffer}, f::Function, idx, loc, VI::AbstractSmoothnessStencil, args...) = inner_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, f, idx, loc, args...)
        @inline inner_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::U{$buffer},  f::Function, idx, loc, VI::AbstractSmoothnessStencil, args...) = inner_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, f, idx, loc, args...)
        @inline inner_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::UZ{$buffer}, f::Function, idx, loc, VI::AbstractSmoothnessStencil, args...) = inner_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, f, idx, loc, args...)

        @inline inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::U{$buffer},  f::Function, idx, loc, VI::AbstractSmoothnessStencil, args...) = inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, f, idx, loc, args...)
        @inline inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::UX{$buffer}, f::Function, idx, loc, VI::AbstractSmoothnessStencil, args...) = inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, f, idx, loc, args...)
        @inline inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::U{$buffer},  f::Function, idx, loc, VI::AbstractSmoothnessStencil, args...) = inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, f, idx, loc, args...)
        @inline inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::UY{$buffer}, f::Function, idx, loc, VI::AbstractSmoothnessStencil, args...) = inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, f, idx, loc, args...)
        @inline inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::U{$buffer},  f::Function, idx, loc, VI::AbstractSmoothnessStencil, args...) = inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, f, idx, loc, args...)
        @inline inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::UZ{$buffer}, f::Function, idx, loc, VI::AbstractSmoothnessStencil, args...) = inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, f, idx, loc, args...)
    end
end
