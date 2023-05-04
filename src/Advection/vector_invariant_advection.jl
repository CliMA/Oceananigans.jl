using Oceananigans.Operators
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, Γᶠᶠᶜ

struct EnergyConservingScheme{FT}    <: AbstractAdvectionScheme{1, FT} end
struct EnstrophyConservingScheme{FT} <: AbstractAdvectionScheme{1, FT} end

EnergyConservingScheme(FT::DataType = Float64)    = EnergyConservingScheme{FT}()
EnstrophyConservingScheme(FT::DataType = Float64) = EnstrophyConservingScheme{FT}()

struct VectorInvariant{N, FT, Z, D, ZS, DS, V} <: AbstractAdvectionScheme{N, FT}
    "reconstruction scheme for vorticity flux"
    vorticity_scheme   :: Z
    "reconstruction scheme for divergence flux"
    divergence_scheme  :: D
    "stencil used for assessing vorticity smoothness"
    vorticity_stencil  :: ZS
    "stencil used for assessing divergence smoothness"
    divergence_stencil :: DS
    "reconstruction scheme for vertical advection"
    vertical_scheme    :: V
    
    function VectorInvariant{N, FT}(vorticity_scheme::Z, divergence_scheme::D, vorticity_stencil::ZS, divergence_stencil::DS, vertical_scheme::V) where {N, FT, Z, D, ZS, DS, V}
        return new{N, FT, Z, D, ZS, DS, V}(vorticity_scheme, divergence_scheme, vorticity_stencil, divergence_stencil, vertical_scheme)
    end
end

"""
    VectorInvariant(; vorticity_scheme::AbstractAdvectionScheme{N, FT} = EnstrophyConservingScheme(), 
                      divergence_scheme  = nothing, 
                      vorticity_stencil  = VelocityStencil(),
                      divergence_stencil = DefaultStencil(),
                      vertical_scheme    = EnergyConservingScheme()) where {N, FT}
               
Construct a vector invariant momentum advection scheme of order `N * 2 - 1`.

Keyword arguments
=================

- `vorticity_scheme`: Scheme used for `Center` reconstruction of vorticity, options are upwind advection schemes
                      - `UpwindBiased` and `WENO` - in addition to an `EnergyConservingScheme` and an `EnstrophyConservingScheme`
                      (defaults to `EnstrophyConservingScheme`)
- `divergence_scheme`: Scheme used for `Face` reconstruction of divergence. Options are upwind advection schemes
                       - `UpwindBiased` and `WENO` - or `nothing`. In case `nothing` is specified, divergence flux is
                       absorbed into the vertical advection term (defaults to `nothing`). If `vertical_scheme` isa `EnergyConservingScheme`,
                       divergence flux is absorbed in vertical advection and this keyword argument has no effect
- `vorticity_stencil`: Stencil used for smoothness indicators in case of a `WENO` upwind reconstruction. Choices are between `VelocityStencil`
                       which uses the horizontal velocity field to diagnose smoothness and `DefaultStencil` which uses the variable
                       being transported (defaults to `VelocityStencil`)
- `divergence_stencil`: same as `vorticity_stencil` but for divergence reconstruction (defaults to `DefaultStencil`)
- `vertical_scheme`: Scheme used for vertical advection of horizontal momentum. It has to be consistent with the choice of 
                     `divergence_stencil`. If the latter is a `Nothing`, only `EnergyConservingScheme` is available (this keyword
                     argument has no effect). In case `divergence_scheme` is an `AbstractUpwindBiasedAdvectionScheme`, 
                     `vertical_scheme` describes a flux form reconstruction of vertical momentum advection, and any 
                     advection scheme can be used - `Centered`, `UpwindBiased` and `WENO` (defaults to `EnergyConservingScheme`)

Examples
========
```jldoctest
julia> using Oceananigans

julia> VectorInvariant()
Vector Invariant reconstruction, maximum order 1 
 Vorticity flux scheme: 
    └── EnstrophyConservingScheme{Float64} 
 Divergence flux scheme: 
    └── Nothing 
 Vertical advection scheme: 
    └── EnergyConservingScheme{Float64}

```
```jldoctest
julia> using Oceananigans

julia> VectorInvariant(vorticity_scheme = WENO(), divergence_scheme = WENO(), vertical_scheme = WENO(order = 3))
Vector Invariant reconstruction, maximum order 5 
 Vorticity flux scheme: 
    └── WENO reconstruction order 5 with smoothness stencil Oceananigans.Advection.VelocityStencil()
 Divergence flux scheme: 
    └── WENO reconstruction order 5 with smoothness stencil Oceananigans.Advection.DefaultStencil()
 Vertical advection scheme: 
    └── WENO reconstruction order 3
```
"""
function VectorInvariant(; vorticity_scheme::AbstractAdvectionScheme{N, FT} = EnstrophyConservingScheme(), 
                           divergence_scheme  = nothing, 
                           vorticity_stencil  = VelocityStencil(),
                           divergence_stencil = DefaultStencil(),
                           vertical_scheme    = EnergyConservingScheme()) where {N, FT}

    divergence_scheme, vertical_scheme = validate_divergence_and_vertical_scheme(divergence_scheme, vertical_scheme)
        
    return VectorInvariant{N, FT}(vorticity_scheme, divergence_scheme, vorticity_stencil, divergence_stencil, vertical_scheme)
end

Base.summary(a::VectorInvariant{N}) where N = string("Vector Invariant reconstruction, maximum order ", N*2-1)

Base.show(io::IO, a::VectorInvariant{N, FT}) where {N, FT} =
    print(io, summary(a), " \n",
              " Vorticity flux scheme: ", "\n",
              "    └── $(summary(a.vorticity_scheme)) $(a.vorticity_scheme isa WENO ? "with smoothness stencil $(a.vorticity_stencil)" : "")\n",
              " Divergence flux scheme: ", "\n",
              "    └── $(summary(a.divergence_scheme)) $(a.divergence_scheme isa WENO ? "with smoothness stencil $(a.divergence_stencil)" : "")\n",
              " Vertical advection scheme: ", "\n",
              "    └── $(summary(a.vertical_scheme))")

# Make sure that divergence is absorbed in the vertical scheme is 1. divergence_schem == Nothing 2. vertical_scheme == EnergyConservingScheme
validate_divergence_and_vertical_scheme(divergence_scheme, vertical_scheme)          = (divergence_scheme, vertical_scheme)
validate_divergence_and_vertical_scheme(::Nothing, vertical_scheme)                  = (nothing, EnergyConservingScheme())
validate_divergence_and_vertical_scheme(::Nothing, ::EnergyConservingScheme)         = (nothing, EnergyConservingScheme())
validate_divergence_and_vertical_scheme(divergence_scheme, ::EnergyConservingScheme) = (nothing, EnergyConservingScheme())

# Since vorticity itself requires one halo, if we use an upwinding scheme (N > 1) we require one additional
# halo for vector invariant advection
required_halo_size(scheme::VectorInvariant{N}) where N = N == 1 ? N : N + 1

Adapt.adapt_structure(to, scheme::VectorInvariant{N, FT}) where {N, FT} =
        VectorInvariant{N, FT}(Adapt.adapt(to, scheme.vorticity_scheme), 
                               Adapt.adapt(to, scheme.divergence_scheme), 
                               Adapt.adapt(to, scheme.vorticity_stencil), 
                               Adapt.adapt(to, scheme.divergence_stencil), 
                               Adapt.adapt(to, scheme.vertical_scheme))

@inline vertical_scheme(scheme::VectorInvariant) = string(nameof(typeof(scheme.vertical_scheme)))

const VectorInvariantEnergyConserving    = VectorInvariant{<:Any, <:Any, <:EnergyConservingScheme}
const VectorInvariantEnstrophyConserving = VectorInvariant{<:Any, <:Any, <:EnstrophyConservingScheme}

const VectorInvariantVerticallyEnergyConserving  = VectorInvariant{<:Any, <:Any, <:Any, Nothing, <:Any, <:Any, <:EnergyConservingScheme}

@inline U_dot_∇u(i, j, k, grid, scheme::VectorInvariant, U) = (
    + horizontal_advection_U(i, j, k, grid, scheme, U.u, U.v)
    + vertical_advection_U(i, j, k, grid, scheme, U.w, U.u)
    + bernoulli_head_U(i, j, k, grid, scheme, U.u, U.v))
    
@inline U_dot_∇v(i, j, k, grid, scheme::VectorInvariant, U) = (
    + horizontal_advection_V(i, j, k, grid, scheme, U.u, U.v)
    + vertical_advection_V(i, j, k, grid, scheme, U.w, U.v)
    + bernoulli_head_V(i, j, k, grid, scheme, U.u, U.v))

#####
##### Kinetic energy gradient (always the same formulation)
#####

@inline ϕ²(i, j, k, grid, ϕ)       = @inbounds ϕ[i, j, k]^2
@inline Khᶜᶜᶜ(i, j, k, grid, u, v) = (ℑxᶜᵃᵃ(i, j, k, grid, ϕ², u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², v)) / 2

@inline bernoulli_head_U(i, j, k, grid, ::VectorInvariant, u, v) = ∂xᶠᶜᶜ(i, j, k, grid, Khᶜᶜᶜ, u, v)
@inline bernoulli_head_V(i, j, k, grid, ::VectorInvariant, u, v) = ∂yᶜᶠᶜ(i, j, k, grid, Khᶜᶜᶜ, u, v)
    
#####
##### Vertical advection (either conservative or flux form when we upwind the divergence transport)
#####

@inline vertical_advection_U(i, j, k, grid, scheme::VectorInvariant, w, u) = 
    1/Vᶠᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, scheme, w, u)

@inline vertical_advection_V(i, j, k, grid, scheme::VectorInvariant, w, v) = 
    1/Vᶜᶠᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, scheme, w, v)

@inline function advective_momentum_flux_Wu(i, j, k, grid, scheme::VectorInvariant, W, u)

    w̃  =    _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.divergence_scheme, Az_qᶜᶜᶠ, W)
    uᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.vertical_scheme, u)
    uᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.vertical_scheme, u)

    return upwind_biased_product(w̃, uᴸ, uᴿ)
end

@inline function advective_momentum_flux_Wv(i, j, k, grid, scheme::VectorInvariant, W, v)

    w̃  =    _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.divergence_scheme, Az_qᶜᶜᶠ, W)
    vᴸ =  _left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.vertical_scheme, v)
    vᴿ = _right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.vertical_scheme, v)

    return upwind_biased_product(w̃, vᴸ, vᴿ)
end

@inbounds ζ₂wᶠᶜᶠ(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶠᶜᶠ(i, j, k, grid, u) 
@inbounds ζ₁wᶜᶠᶠ(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶜᶠᶠ(i, j, k, grid, v) 
        
@inline vertical_advection_U(i, j, k, grid, ::VectorInvariantVerticallyEnergyConserving, w, u) =  ℑzᵃᵃᶜ(i, j, k, grid, ζ₂wᶠᶜᶠ, u, w) / Azᶠᶜᶜ(i, j, k, grid)
@inline vertical_advection_V(i, j, k, grid, ::VectorInvariantVerticallyEnergyConserving, w, v) =  ℑzᵃᵃᶜ(i, j, k, grid, ζ₁wᶜᶠᶠ, v, w) / Azᶜᶠᶜ(i, j, k, grid)

#####
##### Horizontal advection 4 formulations:
#####  1. Energy conservative                (divergence transport absorbed in vertical advection term, vertical advection with EnergyConservingScheme())
#####  2. Enstrophy conservative             (divergence transport absorbed in vertical advection term, vertical advection with EnergyConservingScheme())
#####  3. Vorticity upwinding                (divergence transport absorbed in vertical advection term, vertical advection with EnergyConservingScheme())
#####  4. Vorticity and Divergence upwinding (vertical advection term formulated in flux form, requires an advection scheme other than EnergyConservingScheme)
#####

######
###### Conserving scheme
###### Follows https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#vector-invariant-momentum-equations
######

@inline ζ_ℑx_vᶠᶠᵃ(i, j, k, grid, u, v) = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)
@inline ζ_ℑy_uᶠᶠᵃ(i, j, k, grid, u, v) = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)

@inline horizontal_advection_U(i, j, k, grid, ::VectorInvariantEnergyConserving, u, v) = - ℑyᵃᶜᵃ(i, j, k, grid, ζ_ℑx_vᶠᶠᵃ, u, v) / Δxᶠᶜᶜ(i, j, k, grid)
@inline horizontal_advection_V(i, j, k, grid, ::VectorInvariantEnergyConserving, u, v) = + ℑxᶜᵃᵃ(i, j, k, grid, ζ_ℑy_uᶠᶠᵃ, u, v) / Δyᶜᶠᶜ(i, j, k, grid)

@inline horizontal_advection_U(i, j, k, grid, ::VectorInvariantEnstrophyConserving, u, v) = - ℑyᵃᶜᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 
@inline horizontal_advection_V(i, j, k, grid, ::VectorInvariantEnstrophyConserving, u, v) = + ℑxᶜᵃᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)

######
###### Upwinding schemes
######

const UpwindVorticityVectorInvariant = VectorInvariant{<:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, Nothing}
const UpwindFullVectorInvariant      = VectorInvariant{<:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:AbstractUpwindBiasedAdvectionScheme}

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

## Upwinding `δ` is not like upwinding `ζ`. `u` is trasporting `ζ`, while 
## `u` is the transported quantity for the `δ` term that derives from the vertical advection of `u`
## For this reason, the divergence (`δ`) must be multiplied by the area in z to account for the flux of `w` 
## (as done for the other advecting velocities). 

@inline function horizontal_advection_U(i, j, k, grid, scheme::UpwindFullVectorInvariant, u, v)
    
    Sζ = scheme.vorticity_stencil

    @inbounds v̂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 
    ζᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v) 
    ζᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v) 

    Sδ = scheme.divergence_stencil
    
    @inbounds û = u[i, j, k]
    δᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.divergence_scheme, flux_div_xyᶜᶜᶜ, Sδ, u, v) / Azᶠᶜᶜ(i, j, k, grid)
    δᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.divergence_scheme, flux_div_xyᶜᶜᶜ, Sδ, u, v) / Azᶠᶜᶜ(i, j, k, grid)

    return - upwind_biased_product(v̂, ζᴸ, ζᴿ) + upwind_biased_product(û, δᴸ, δᴿ) 
end

@inline function horizontal_advection_V(i, j, k, grid, scheme::UpwindFullVectorInvariant, u, v) 

    Sζ = scheme.vorticity_stencil

    @inbounds û  =  ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)
    ζᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v) 
    ζᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v) 

    Sδ = scheme.divergence_stencil

    @inbounds v̂ = v[i, j, k]
    δᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.divergence_scheme, flux_div_xyᶜᶜᶜ, Sδ, u, v) / Azᶜᶠᶜ(i, j, k, grid)
    δᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.divergence_scheme, flux_div_xyᶜᶜᶜ, Sδ, u, v) / Azᶜᶠᶜ(i, j, k, grid)

    return upwind_biased_product(û, ζᴸ, ζᴿ) + upwind_biased_product(v̂, δᴸ, δᴿ) 
end

######
###### Conservative formulation of momentum advection
######

@inline U_dot_∇u(i, j, k, grid, scheme::AbstractAdvectionScheme, U) = div_𝐯u(i, j, k, grid, scheme, U, U.u)
@inline U_dot_∇v(i, j, k, grid, scheme::AbstractAdvectionScheme, U) = div_𝐯v(i, j, k, grid, scheme, U, U.v)

######
###### No advection
######

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
