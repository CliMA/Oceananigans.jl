using Oceananigans.Operators

struct EnergyConservingScheme{FT}    <: AbstractAdvectionScheme{1, FT} end
struct EnstrophyConservingScheme{FT} <: AbstractAdvectionScheme{1, FT} end

EnergyConservingScheme(FT::DataType = Float64)    = EnergyConservingScheme{FT}()
EnstrophyConservingScheme(FT::DataType = Float64) = EnstrophyConservingScheme{FT}()

struct VectorInvariant{N, FT, Z, D, ZS, DS, V} <: AbstractAdvectionScheme{N, FT}
    vorticity_scheme   :: Z
    divergence_scheme  :: D
    vorticity_stencil  :: ZS
    divergence_stencil :: DS
    vertical_scheme    :: V
    
    function VectorInvariant{N, FT}(vorticity_scheme::Z, divergence_scheme::D, vorticity_stencil::ZS, divergence_stencil::DS, vertical_scheme::V) where {N, FT, Z, D, ZS, DS, V}
        return new{N, FT, Z, D, ZS, DS, V}(vorticity_scheme, divergence_scheme, vorticity_stencil, divergence_stencil, vertical_scheme)
    end
end

function VectorInvariant(; vorticity_scheme::AbstractAdvectionScheme{N, FT} = EnstrophyConservingScheme(), 
                           divergence_scheme  = nothing, 
                           vorticity_stencil  = VelocityStencil(),
                           divergence_stencil = DefaultStencil(),
                           vertical_scheme    = EnergyConservingScheme()) where {N, FT}

    divergence_scheme, vertical_scheme = validate_divergence_and_vertical_scheme(divergence_scheme, vertical_scheme)

    divergence_scheme isa Nothing && @warn "Using a fully conservative vector invariant scheme, divergence transport is absorbed in the vertical advection"
        
    return VectorInvariant{N, FT}(vorticity_scheme, divergence_scheme, vorticity_stencil, divergence_stencil, vertical_scheme)
end

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

const VectorInvariantConserving = Union{VectorInvariantEnergyConserving, VectorInvariantEnstrophyConserving}

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
    1/Vᶠᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, scheme.vertical_scheme, w, u)

@inline vertical_advection_V(i, j, k, grid, scheme::VectorInvariant, w, v) = 
    1/Vᶜᶠᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, scheme.vertical_scheme, w, v)

@inbounds ζ₂wᶠᶜᶠ(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶠᶜᶠ(i, j, k, grid, u) 
@inbounds ζ₁wᶜᶠᶠ(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶜᶠᶠ(i, j, k, grid, v) 
        
@inline vertical_advection_U(i, j, k, grid, ::VectorInvariantConserving, w, u) =  ℑzᵃᵃᶜ(i, j, k, grid, ζ₂wᶠᶜᶠ, u, w) / Azᶠᶜᶜ(i, j, k, grid)
@inline vertical_advection_V(i, j, k, grid, ::VectorInvariantConserving, w, v) =  ℑzᵃᵃᶜ(i, j, k, grid, ζ₁wᶜᶠᶠ, v, w) / Azᶜᶠᶜ(i, j, k, grid)

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

@inline function horizontal_advection_U(i, j, k, grid, scheme::UpwindFullVectorInvariant, u, v)
    
    Sζ = scheme.vorticity_stencil

    @inbounds v̂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 
    ζᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v)
    ζᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v)

    Sδ = scheme.divergence_stencil
    
    @inbounds û = u[i, j, k]
    δᴸ =  _left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.divergence_scheme, div_xyᶜᶜᶜ, Sδ, u, v)
    δᴿ = _right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.divergence_scheme, div_xyᶜᶜᶜ, Sδ, u, v)

    return upwind_biased_product(û, δᴸ, δᴿ) - upwind_biased_product(v̂, ζᴸ, ζᴿ)
end

@inline function horizontal_advection_V(i, j, k, grid, scheme::UpwindFullVectorInvariant, u, v) 

    Sζ = scheme.vorticity_stencil

    @inbounds û  =  ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)
    ζᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v)
    ζᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, Sζ, u, v)

    Sδ = scheme.divergence_stencil

    @inbounds v̂ = v[i, j, k]
    δᴸ =  _left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.divergence_scheme, div_xyᶜᶜᶜ, Sδ, u, v)
    δᴿ = _right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.divergence_scheme, div_xyᶜᶜᶜ, Sδ, u, v)

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