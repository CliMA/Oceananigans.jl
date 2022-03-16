using Oceananigans.Operators

struct EnergyConservingScheme end
struct EnstrophyConservingScheme end

struct VectorInvariant{S}
    scheme :: S
end

VectorInvariant(; scheme::S = EnergyConservingScheme()) where S = VectorInvariant{S}(scheme)

const VectorInvariantSchemes = Union{VectorInvariant, WENOVectorInvariant}

const VectorInvariantEnergyConserving = VectorInvariant{<:EnergyConservingScheme}
const VectorInvariantEnstrophyConserving = VectorInvariant{<:EnstrophyConservingScheme}

######
###### Horizontally-vector-invariant formulation of momentum scheme
######
###### Follows https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#vector-invariant-momentum-equations
######

@inline U_dot_∇u(i, j, k, grid, scheme::VectorInvariantSchemes, U) = (
    + vertical_vorticity_U(i, j, k, grid, scheme, U.u, U.v)  # Vertical relative vorticity term
    + vertical_advection_U(i, j, k, grid, scheme, U.u, U.w)  # Horizontal vorticity / vertical advection term
    + bernoulli_head_U(i, j, k, grid, scheme, U.u, U.v))     # Bernoulli head term
    
@inline U_dot_∇v(i, j, k, grid, scheme::VectorInvariantSchemes, U) = (
    + vertical_vorticity_V(i, j, k, grid, scheme, U.u, U.v)  # Vertical relative vorticity term
    + vertical_advection_V(i, j, k, grid, scheme, U.v, U.w)  # Horizontal vorticity / vertical advection term
    + bernoulli_head_V(i, j, k, grid, scheme, U.u, U.v))     # Bernoulli head term

####
#### Bernoulli head terms
####

@inline bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantSchemes, u, v) = ∂xᶠᶜᶜ(i, j, k, grid, Khᶜᶜᶜ, scheme, u, v)    
@inline bernoulli_head_V(i, j, k, grid, scheme::VectorInvariantSchemes, u, v) = ∂yᶜᶠᶜ(i, j, k, grid, Khᶜᶜᶜ, scheme, u, v)  

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2
@inline Khᶜᶜᶜ(i, j, k, grid, ::VectorInvariantSchemes, u, v) = (ℑxᶜᵃᵃ(i, j, k, grid, ϕ², u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², v)) / 2

####
#### Horizontal advection terms
####

@inline ζ_ℑx_vᶠᶠᵃ(i, j, k, grid, u, v) = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)
@inline ζ_ℑy_uᶠᶠᵃ(i, j, k, grid, u, v) = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)

@inline vertical_vorticity_U(i, j, k, grid, ::VectorInvariantEnergyConserving, u, v) = - ℑyᵃᶜᵃ(i, j, k, grid, ζ_ℑx_vᶠᶠᵃ, u, v) / Δxᶠᶜᶜ(i, j, k, grid)
@inline vertical_vorticity_V(i, j, k, grid, ::VectorInvariantEnergyConserving, u, v) = + ℑxᶜᵃᵃ(i, j, k, grid, ζ_ℑy_uᶠᶠᵃ, u, v) / Δyᶜᶠᶜ(i, j, k, grid)

@inline vertical_vorticity_U(i, j, k, grid, ::VectorInvariantEnstrophyConserving, u, v) = - ℑyᵃᶜᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 
@inline vertical_vorticity_V(i, j, k, grid, ::VectorInvariantEnstrophyConserving, u, v) = + ℑxᶜᵃᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)

@inline function vertical_vorticity_U(i, j, k, grid, scheme::WENOVectorInvariant{FT, XT, YT, ZT, XS, YS, ZS, VI}, u, v) where {FT, XT, YT, ZT, XS, YS, ZS, VI}
    v̂  =  ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 
    ζᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, ζ₃ᶠᶠᶜ, VI, u, v)
    ζᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, ζ₃ᶠᶠᶜ, VI, u, v)
    return - upwind_biased_product(v̂, ζᴸ, ζᴿ) 
end

@inline function vertical_vorticity_V(i, j, k, grid, scheme::WENOVectorInvariant{FT, XT, YT, ZT, XS, YS, ZS, VI}, u, v) where {FT, XT, YT, ZT, XS, YS, ZS, VI}
    û  =  ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)
    ζᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, ζ₃ᶠᶠᶜ, VI, u, v)
    ζᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, ζ₃ᶠᶠᶜ, VI, u, v)
    return + upwind_biased_product(û, ζᴸ, ζᴿ) 
end

####
#### Vertical advection terms
####

@inbounds ζ₂wᶠᶜᶠ(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶠᶜᶠ(i, j, k, grid, u) / Azᶠᶜᶜ(i, j, k, grid)
@inbounds ζ₁wᶜᶠᶠ(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶜᶠᶠ(i, j, k, grid, v) / Azᶜᶠᶜ(i, j, k, grid)
    
@inline vertical_advection_U(i, j, k, grid, ::VectorInvariantSchemes, u, w) =  ℑzᵃᵃᶜ(i, j, k, grid, ζ₂wᶠᶜᶠ, u, w)
@inline vertical_advection_V(i, j, k, grid, ::VectorInvariantSchemes, v, w) =  ℑzᵃᵃᶜ(i, j, k, grid, ζ₁wᶜᶠᶠ, v, w)

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
