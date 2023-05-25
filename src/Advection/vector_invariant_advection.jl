using Oceananigans.Operators
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, Γᶠᶠᶜ

struct EnergyConservingScheme{FT}    <: AbstractAdvectionScheme{1, FT} end
struct EnstrophyConservingScheme{FT} <: AbstractAdvectionScheme{1, FT} end

abstract type AbstractUpwindingTreatment end

struct FullUpwinding    <: AbstractUpwindingTreatment end
struct PartialUpwinding <: AbstractUpwindingTreatment end
struct SplitUpwinding   <: AbstractUpwindingTreatment end

EnergyConservingScheme(FT::DataType = Float64)    = EnergyConservingScheme{FT}()
EnstrophyConservingScheme(FT::DataType = Float64) = EnstrophyConservingScheme{FT}()

struct VectorInvariant{N, FT, Z, ZS, V, D, US, VS, M} <: AbstractAdvectionScheme{N, FT}
    "reconstruction scheme for vorticity flux"
    vorticity_scheme     :: Z
    "stencil used for assessing vorticity smoothness"
    vorticity_stencil    :: ZS
    "stencil used for assessing divergence smoothness"
    vertical_scheme      :: V
    "treatment of upwinding"
    upwinding_treatment  :: D
    "stencil used for assessing u-derivative smoothness"
    u_stencil            :: US
    "stencil used for assessing v-derivative smoothness"
    v_stencil            :: VS
    
    function VectorInvariant{N, FT, M}(vorticity_scheme::Z, vorticity_stencil::ZS, vertical_scheme::V, upwinding_treatment::D, u_stencil::US, v_stencil::VS) where {N, FT, Z, ZS, V, D, US, VS, M}
        return new{N, FT, Z, ZS, V, D, US, VS, M}(vorticity_scheme, vorticity_stencil, vertical_scheme, upwinding_treatment, u_stencil, v_stencil)
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
- `u_stencil`: Stencil used for smoothness indicators of `δ_U` in case of a `WENO` upwind reconstruction. Choices are between `DefaultStencil` 
               which uses the variable being transported, or `FunctionStencil(smoothness_function)` where `smoothness_function` is a 
               custom function (defaults to `FunctionStencil(divergence_smoothness)`)
- `v_stencil`: Same as `u_stencil` but for the smoothness of `δ_V`
- `multi_dimensional_stencil` : if true, use a horizontal two dimensional stencil for the reconstruction of vorticity and divergence.
                                The tangential (not upwinded) direction is treated with a 5th order centered WENO reconstruction

Examples
========
```jldoctest
julia> using Oceananigans

julia> VectorInvariant()
Vector Invariant, Dimension-by-dimension reconstruction 
 Vorticity flux scheme: 
    └── EnstrophyConservingScheme{Float64}
 Vertical advection / Divergence flux scheme: 
    └── EnergyConservingScheme{Float64}

```
```jldoctest
julia> using Oceananigans

julia> VectorInvariant(vorticity_scheme = WENO(), vertical_scheme = WENO(order = 3))
Vector Invariant, Dimension-by-dimension reconstruction 
 Vorticity flux scheme: 
    └── WENO reconstruction order 5
      └── smoothness ζ: Oceananigans.Advection.VelocityStencil()
 Vertical advection / Divergence flux scheme: 
    └── WENO reconstruction order 3
      └── upwinding treatment: Oceananigans.Advection.FullUpwinding()
      └── smoothness u: FunctionStencil f = divergence_smoothness 
      └── smoothness v: FunctionStencil f = divergence_smoothness
```
"""
function VectorInvariant(; vorticity_scheme::AbstractAdvectionScheme{N, FT} = EnstrophyConservingScheme(), 
                           vorticity_stencil    = VelocityStencil(),
                           vertical_scheme      = EnergyConservingScheme(),
                           upwinding_treatment  = FullUpwinding(),
                           u_stencil            = FunctionStencil(divergence_smoothness),
                           v_stencil            = FunctionStencil(divergence_smoothness),
                           multi_dimensional_stencil = false) where {N, FT}
        
    return VectorInvariant{N, FT, multi_dimensional_stencil}(vorticity_scheme, vorticity_stencil, vertical_scheme, upwinding_treatment, u_stencil, v_stencil)
end

const VectorInvariantEnergyConserving            = VectorInvariant{<:Any, <:Any, <:EnergyConservingScheme}
const VectorInvariantEnstrophyConserving         = VectorInvariant{<:Any, <:Any, <:EnstrophyConservingScheme}
const VectorInvariantVerticallyEnergyConserving  = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:EnergyConservingScheme}

const VectorInvariantUpwindVorticity  = VectorInvariant{<:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme}
const MultiDimensionalVectorInvariant = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, true}

Base.summary(a::VectorInvariant)                 = string("Vector Invariant, Dimension-by-dimension reconstruction")
Base.summary(a::MultiDimensionalVectorInvariant) = string("Vector Invariant, Multidimensional reconstruction")

Base.show(io::IO, a::VectorInvariant{N, FT}) where {N, FT} =
    print(io, summary(a), " \n",
              " Vorticity flux scheme: ", "\n",
              "    └── $(summary(a.vorticity_scheme))",
              "$(a.vorticity_scheme isa WENO ? "\n      └── smoothness ζ: $(a.vorticity_stencil)\n" : "\n")",
              " Vertical advection / Divergence flux scheme: ", "\n",
              "    └── $(summary(a.vertical_scheme))",
              "$(a.vertical_scheme isa AbstractUpwindBiasedAdvectionScheme ? "\n      └── upwinding treatment: $(a.upwinding_treatment)" : "")",
              "$(a.vertical_scheme isa WENO ? "\n      └── smoothness u: $(a.u_stencil) \n      └── smoothness v: $(a.v_stencil)" : "")")

# Since vorticity itself requires one halo, if we use an upwinding scheme (N > 1) we require one additional
# halo for vector invariant advection
required_halo_size(scheme::VectorInvariant{N}) where N = N == 1 ? N : N + 1

Adapt.adapt_structure(to, scheme::VectorInvariant{N, FT, Z, ZS, V, D, US, VS, M}) where {N, FT, Z, ZS, V, D, US, VS, M} =
        VectorInvariant{N, FT, M}(Adapt.adapt(to, scheme.vorticity_scheme), 
                                  Adapt.adapt(to, scheme.vorticity_stencil), 
                                  Adapt.adapt(to, scheme.vertical_scheme),
                                  Adapt.adapt(to, scheme.upwinding_treatment),
                                  Adapt.adapt(to, scheme.u_stencil),
                                  Adapt.adapt(to, scheme.v_stencil))

@inline U_dot_∇u(i, j, k, grid, scheme::VectorInvariant, U) = (
    + horizontal_advection_U(i, j, k, grid, scheme, U.u, U.v)
    + vertical_advection_U(i, j, k, grid, scheme, U.w, U.u, U.v)
    + bernoulli_head_U(i, j, k, grid, scheme, U.u, U.v))
    
@inline U_dot_∇v(i, j, k, grid, scheme::VectorInvariant, U) = (
    + horizontal_advection_V(i, j, k, grid, scheme, U.u, U.v)
    + vertical_advection_V(i, j, k, grid, scheme, U.w, U.u, U.v)
    + bernoulli_head_V(i, j, k, grid, scheme, U.u, U.v))

# Extend interpolate functions for VectorInvariant to allow MultiDimensional reconstruction
for bias in (:_left_biased, :_right_biased, :_symmetric)
    for (dir1, dir2) in zip((:xᶠᵃᵃ, :xᶜᵃᵃ, :yᵃᶠᵃ, :yᵃᶜᵃ), (:y, :y, :x, :x))
        interp_func = Symbol(bias, :_interpolate_, dir1)
        multidim_interp   = Symbol(:_multi_dimensional_reconstruction_, dir2)

        @eval begin
            $interp_func(i, j, k, grid, ::VectorInvariant, interp_scheme, args...) = 
                    $interp_func(i, j, k, grid, interp_scheme, args...)
            $interp_func(i, j, k, grid, ::MultiDimensionalVectorInvariant, interp_scheme, args...) = 
                    $multidim_interp(i, j, k, grid, interp_scheme, $interp_func, args...)
        end
    end
end

#####
#####  Vertical advection + Kinetic Energy gradient. 3 Formulations:
#####  1. Energy conserving
#####  2. Dimension-By-Dimension Divergence upwinding (Partial, Split or Full)
#####  3. Multi-Dimensional Divergence upwinding      (Partial, Split or Full)
#####

#####
##### Conservative Kinetic Energy Gradient (1)
#####

@inline ϕ²(i, j, k, grid, ϕ)       = @inbounds ϕ[i, j, k]^2
@inline Khᶜᶜᶜ(i, j, k, grid, u, v) = (ℑxᶜᵃᵃ(i, j, k, grid, ϕ², u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², v)) / 2

@inline bernoulli_head_U(i, j, k, grid, scheme, u, v) = ∂xᶠᶜᶜ(i, j, k, grid, Khᶜᶜᶜ, u, v)
@inline bernoulli_head_V(i, j, k, grid, scheme, u, v) = ∂yᶜᶠᶜ(i, j, k, grid, Khᶜᶜᶜ, u, v)

#####
##### Conservative vertical advection 
##### Follows https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#vector-invariant-momentum-equations
#####

@inbounds ζ₂wᶠᶜᶠ(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶠᶜᶠ(i, j, k, grid, u) 
@inbounds ζ₁wᶜᶠᶠ(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶜᶠᶠ(i, j, k, grid, v) 

@inline vertical_advection_U(i, j, k, grid, ::VectorInvariantVerticallyEnergyConserving, w, u, v) =  ℑzᵃᵃᶜ(i, j, k, grid, ζ₂wᶠᶜᶠ, u, w) / Azᶠᶜᶜ(i, j, k, grid)
@inline vertical_advection_V(i, j, k, grid, ::VectorInvariantVerticallyEnergyConserving, w, u, v) =  ℑzᵃᵃᶜ(i, j, k, grid, ζ₁wᶜᶠᶠ, v, w) / Azᶜᶠᶜ(i, j, k, grid)

#####
##### Upwinding vertical advection (2. and 3.)
#####

@inline function vertical_advection_U(i, j, k, grid, scheme::VectorInvariant, w, u, v) 
    
    Φᵟ = upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme, u, v)
    𝒜ᶻ = δzᵃᵃᶜ(i, j, k, grid, advective_momentum_flux_Wu, scheme.vertical_scheme, w, u)

    return 1/Vᶠᶜᶜ(i, j, k, grid) * (Φᵟ + 𝒜ᶻ)
end

@inline function vertical_advection_V(i, j, k, grid, scheme::VectorInvariant, w, u, v) 

    Φᵟ = upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme, u, v)
    𝒜ᶻ = δzᵃᵃᶜ(i, j, k, grid, advective_momentum_flux_Wv, scheme.vertical_scheme, w, v)

    return 1/Vᶜᶠᶜ(i, j, k, grid) * (Φᵟ + 𝒜ᶻ)
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

@inline function horizontal_advection_U(i, j, k, grid, scheme::VectorInvariantUpwindVorticity, u, v)
    
    Sζ = scheme.vorticity_stencil

    @inbounds v̂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 
    ζᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v)
    ζᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v)

    return - upwind_biased_product(v̂, ζᴸ, ζᴿ)
end

@inline function horizontal_advection_V(i, j, k, grid, scheme::VectorInvariantUpwindVorticity, u, v) 

    Sζ = scheme.vorticity_stencil

    @inbounds û  =  ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)
    ζᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v)
    ζᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v)

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
