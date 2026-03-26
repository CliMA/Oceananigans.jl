# These are also used in Coriolis/hydrostatic_spherical_coriolis.jl
struct EnergyConserving{FT}    <: AbstractAdvectionScheme{1, FT} end
struct EnstrophyConserving{FT} <: AbstractAdvectionScheme{1, FT} end

EnergyConserving(FT::DataType = Oceananigans.defaults.FloatType)    = EnergyConserving{FT}()
EnstrophyConserving(FT::DataType = Oceananigans.defaults.FloatType) = EnstrophyConserving{FT}()

struct VectorInvariant{N, FT, M, Z, ZS, V, K, D, U} <: AbstractAdvectionScheme{N, FT}
    vorticity_scheme               :: Z  # reconstruction scheme for vorticity flux
    vorticity_stencil              :: ZS # stencil used for assessing vorticity smoothness
    vertical_advection_scheme      :: V  # reconstruction scheme for vertical advection
    kinetic_energy_gradient_scheme :: K  # reconstruction scheme for kinetic energy gradient
    divergence_scheme              :: D  # reconstruction scheme for divergence flux
    upwinding                      :: U  # treatment of upwinding for divergence flux and kinetic energy gradient

    function VectorInvariant{N, FT, M}(vorticity_scheme::Z,
                                       vorticity_stencil::ZS,
                                       vertical_advection_scheme::V,
                                       kinetic_energy_gradient_scheme::K,
                                       divergence_scheme::D,
                                       upwinding::U) where {N, FT, M, Z, ZS, V, K, D, U}

        return new{N, FT, M, Z, ZS, V, K, D, U}(vorticity_scheme,
                                                vorticity_stencil,
                                                vertical_advection_scheme,
                                                kinetic_energy_gradient_scheme,
                                                divergence_scheme,
                                                upwinding)
    end
end

"""
    VectorInvariant(; vorticity_scheme = EnstrophyConserving(),
                      vorticity_stencil = VelocityStencil(),
                      vertical_advection_scheme = EnergyConserving(),
                      divergence_scheme = vertical_advection_scheme,
                      kinetic_energy_gradient_scheme = divergence_scheme,
                      upwinding  = OnlySelfUpwinding(; cross_scheme = divergence_scheme),
                      multi_dimensional_stencil = false)

Return a vector-invariant momentum advection scheme.

Keyword arguments
=================

- `vorticity_scheme`: Scheme used for `Center` reconstruction of vorticity. Default: `EnstrophyConserving()`. Options:
  * `UpwindBiased()`
  * `WENO()`
  * `EnergyConserving()`
  * `EnstrophyConserving()`

- `vorticity_stencil`: Stencil used for smoothness indicators for `WENO` schemes. Default: `VelocityStencil()`. Options:
  * `VelocityStencil()` (smoothness based on horizontal velocities)
  * `DefaultStencil()` (smoothness based on variable being reconstructed)

- `vertical_advection_scheme`: Scheme used for vertical advection of horizontal momentum. Default: `EnergyConserving()`.

- `kinetic_energy_gradient_scheme`: Scheme used for kinetic energy gradient reconstruction. Default: `vertical_advection_scheme`.

- `divergence_scheme`: Scheme used for divergence flux. Only upwinding schemes are supported. Default: `vorticity_scheme`.

- `upwinding`: Treatment of upwinded reconstruction of divergence and kinetic energy gradient. Default: `OnlySelfUpwinding()`. Options:
  * `CrossAndSelfUpwinding()`
  * `OnlySelfUpwinding()`

- `multi_dimensional_stencil`: whether or not to use a horizontal two-dimensional stencil for the reconstruction
                               of vorticity, divergence, and kinetic energy gradient. Currently the "tangential"
                               direction uses 5th-order centered WENO reconstruction. Default: false

Examples
========

```jldoctest vector_invariant
julia> using Oceananigans

julia> VectorInvariant()
VectorInvariant
├── vorticity_scheme: Oceananigans.Advection.EnstrophyConserving{Float64}
└── vertical_advection_scheme: Oceananigans.Advection.EnergyConserving{Float64}
```
"""
function VectorInvariant(FT = Oceananigans.defaults.FloatType;
                         vorticity_scheme = EnstrophyConserving(FT),
                         vorticity_stencil = VelocityStencil(),
                         vertical_advection_scheme = EnergyConserving(FT),
                         divergence_scheme = vertical_advection_scheme,
                         kinetic_energy_gradient_scheme = divergence_scheme,
                         upwinding = OnlySelfUpwinding(; cross_scheme = divergence_scheme),
                         multi_dimensional_stencil = false)

    N = max(required_halo_size_x(vorticity_scheme),
            required_halo_size_y(vorticity_scheme),
            required_halo_size_x(divergence_scheme),
            required_halo_size_y(divergence_scheme),
            required_halo_size_x(kinetic_energy_gradient_scheme),
            required_halo_size_y(kinetic_energy_gradient_scheme),
            required_halo_size_z(vertical_advection_scheme))

    FT = eltype(vorticity_scheme)

    return VectorInvariant{N, FT, multi_dimensional_stencil}(vorticity_scheme,
                                                             vorticity_stencil,
                                                             vertical_advection_scheme,
                                                             kinetic_energy_gradient_scheme,
                                                             divergence_scheme,
                                                             upwinding)
end

#                                                                 buffer eltype
#                                                 VectorInvariant{N,     FT,    M (multi-dimensionality)
const MultiDimensionalVectorInvariant           = VectorInvariant{<:Any, <:Any, true}

#                                                 VectorInvariant{N,     FT,    M,     Z (vorticity scheme)
const VectorInvariantEnergyConserving           = VectorInvariant{<:Any, <:Any, <:Any, <:EnergyConserving}
const VectorInvariantEnstrophyConserving        = VectorInvariant{<:Any, <:Any, <:Any, <:EnstrophyConserving}
const VectorInvariantUpwindVorticity            = VectorInvariant{<:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme}

#                                                 VectorInvariant{N,     FT,    M,     Z,     ZS,    V (vertical scheme)
const VectorInvariantVerticalEnergyConserving   = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:Any, <:EnergyConserving}

#                                                 VectorInvariant{N,     FT,    M,     Z,     ZS,    V,     K (kinetic energy gradient scheme)
const VectorInvariantKEGradientEnergyConserving = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:EnergyConserving}
const VectorInvariantKineticEnergyUpwinding     = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme}


#                                                 VectorInvariant{N,     FT,    M,     Z,     ZS,     V,     K,     D,                                     U (upwinding)
const VectorInvariantCrossVerticalUpwinding     = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:Any,  <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:CrossAndSelfUpwinding}
const VectorInvariantSelfVerticalUpwinding      = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:Any,  <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:OnlySelfUpwinding}

const WENOVectorInvariant{N} = VectorInvariant{<:Any, <:Any, <:Any, <:WENO{N}} where N

Base.summary(a::VectorInvariant)                 = "VectorInvariant"
Base.summary(a::MultiDimensionalVectorInvariant) = "VectorInvariant, multidimensional reconstruction"

function Base.summary(a::WENOVectorInvariant{N}) where N
    vorticity_order = weno_order(a.vorticity_scheme)
    vertical_order = weno_order(a.vertical_advection_scheme)
    order = weno_order(a.vorticity_scheme)
    FT = eltype(a.vorticity_scheme)
    FT2 = eltype2(a.vorticity_scheme)
    return string("WENOVectorInvariant{$N, $FT, $FT2}(vorticity_order=$vorticity_order, vertical_order=$vertical_order)")
end

function Base.show(io::IO, a::VectorInvariant{N, FT}) where {N, FT}
    print(io, summary(a), '\n')
    print(io, "├── vorticity_scheme: ", summary(a.vorticity_scheme), '\n')

    if a.vorticity_scheme isa WENO
        print(io, "├── vorticity_stencil: ", summary(a.vorticity_stencil), '\n')
    end

    if a.vertical_advection_scheme isa AbstractUpwindBiasedAdvectionScheme
        print(io, "├── vertical_advection_scheme: ", summary(a.vertical_advection_scheme), '\n')
        print(io, "├── kinetic_energy_gradient_scheme: ", summary(a.kinetic_energy_gradient_scheme), '\n')
        print(io, "├── divergence_scheme: ", summary(a.divergence_scheme), '\n')
        print(io, "└── upwinding: ", summary(a.upwinding))
    else
        print(io, "└── vertical_advection_scheme: ", summary(a.vertical_advection_scheme))
    end
end

#####
##### Convenience for WENO Vector Invariant
#####

nothing_to_default(user_value; default=nothing) = isnothing(user_value) ? default : user_value

"""
    WENOVectorInvariant(FT = Float64;
                        upwinding = nothing,
                        vorticity_stencil = VelocityStencil(),
                        order = nothing,
                        vorticity_order = nothing,
                        vertical_order = nothing,
                        divergence_order = nothing,
                        kinetic_energy_gradient_order = nothing,
                        multi_dimensional_stencil = false,
                        minimum_buffer_upwind_order = 1,
                        weno_kw...)

Return a vector-invariant weighted essentially non-oscillatory (WENO) scheme.
See [`VectorInvariant`](@ref) and [`WENO`](@ref) for kwargs definitions.

If `multi_dimensional_stencil = true` is selected, then a 2D horizontal stencil
is implemented for the WENO scheme (instead of a 1D stencil). This 2D horizontal
stencil performs a centered 5th-order WENO reconstruction of vorticity,
divergence and kinetic energy in the horizontal direction tangential to the upwind direction.

Example
=======

```jldoctest weno_vector_invariant
julia> using Oceananigans

julia> WENOVectorInvariant()
WENOVectorInvariant{5, Float64, Float32}(vorticity_order=9, vertical_order=5)
├── vorticity_scheme: WENO{5, Float64, Float32}(order=9)
├── vorticity_stencil: Oceananigans.Advection.VelocityStencil
├── vertical_advection_scheme: WENO{3, Float64, Float32}(order=5)
├── kinetic_energy_gradient_scheme: WENO{3, Float64, Float32}(order=5)
├── divergence_scheme: WENO{3, Float64, Float32}(order=5)
└── upwinding: OnlySelfUpwinding
```
"""
function WENOVectorInvariant(FT::DataType = Oceananigans.defaults.FloatType;
                             upwinding = nothing,
                             vorticity_stencil = VelocityStencil(),
                             order = nothing,
                             vorticity_order = nothing,
                             vertical_order = nothing,
                             divergence_order = nothing,
                             kinetic_energy_gradient_order = nothing,
                             multi_dimensional_stencil = false,
                             minimum_buffer_upwind_order = 1,
                             weno_kw...)

    if isnothing(order) # apply global defaults
        vorticity_order               = nothing_to_default(vorticity_order,  default = 9)
        vertical_order                = nothing_to_default(vertical_order,   default = 5)
        divergence_order              = nothing_to_default(divergence_order, default = 5)
        kinetic_energy_gradient_order = nothing_to_default(kinetic_energy_gradient_order, default = 5)
    else # apply user supplied `order` unless overridden by more specific value
        vorticity_order               = nothing_to_default(vorticity_order,  default = order)
        vertical_order                = nothing_to_default(vertical_order,   default = order)
        divergence_order              = nothing_to_default(divergence_order, default = order)
        kinetic_energy_gradient_order = nothing_to_default(kinetic_energy_gradient_order, default = order)
    end

    vorticity_scheme               = WENO(FT; order=vorticity_order, minimum_buffer_upwind_order, weno_kw...)
    vertical_advection_scheme      = WENO(FT; order=vertical_order, minimum_buffer_upwind_order, weno_kw...)
    kinetic_energy_gradient_scheme = WENO(FT; order=kinetic_energy_gradient_order, minimum_buffer_upwind_order, weno_kw...)
    divergence_scheme              = WENO(FT; order=divergence_order, minimum_buffer_upwind_order, weno_kw...)

    default_upwinding = OnlySelfUpwinding(cross_scheme = divergence_scheme)
    upwinding = nothing_to_default(upwinding; default = default_upwinding)

    schemes = (vorticity_scheme, vertical_advection_scheme, kinetic_energy_gradient_scheme, divergence_scheme)
    NX = maximum(required_halo_size_x(s) for s in schemes)
    NY = maximum(required_halo_size_y(s) for s in schemes)
    NZ = maximum(required_halo_size_z(s) for s in schemes)
    N = max(NX, NY, NZ)

    FT = eltype(vorticity_scheme) # assumption

    return VectorInvariant{N, FT, multi_dimensional_stencil}(vorticity_scheme,
                                                             vorticity_stencil,
                                                             vertical_advection_scheme,
                                                             kinetic_energy_gradient_scheme,
                                                             divergence_scheme,
                                                             upwinding)
end

# Since vorticity itself requires one halo, if we use an upwinding scheme (N > 1) we require one additional
# halo for vector invariant advection
@inline function required_halo_size_x(scheme::VectorInvariant)
    Hx₁ = required_halo_size_x(scheme.vorticity_scheme)
    Hx₂ = required_halo_size_x(scheme.divergence_scheme)
    Hx₃ = required_halo_size_x(scheme.kinetic_energy_gradient_scheme)

    Hx = max(Hx₁, Hx₂, Hx₃)
    return Hx == 1 ? Hx : Hx + 1
end

@inline required_halo_size_y(scheme::VectorInvariant) = required_halo_size_x(scheme)
@inline required_halo_size_z(scheme::VectorInvariant) = required_halo_size_z(scheme.vertical_advection_scheme)

Adapt.adapt_structure(to, scheme::VectorInvariant{N, FT, M}) where {N, FT, M} =
    VectorInvariant{N, FT, M}(Adapt.adapt(to, scheme.vorticity_scheme),
                              Adapt.adapt(to, scheme.vorticity_stencil),
                              Adapt.adapt(to, scheme.vertical_advection_scheme),
                              Adapt.adapt(to, scheme.kinetic_energy_gradient_scheme),
                              Adapt.adapt(to, scheme.divergence_scheme),
                              Adapt.adapt(to, scheme.upwinding))

on_architecture(to, scheme::VectorInvariant{N, FT, M}) where {N, FT, M} =
    VectorInvariant{N, FT, M}(on_architecture(to, scheme.vorticity_scheme),
                              on_architecture(to, scheme.vorticity_stencil),
                              on_architecture(to, scheme.vertical_advection_scheme),
                              on_architecture(to, scheme.kinetic_energy_gradient_scheme),
                              on_architecture(to, scheme.divergence_scheme),
                              on_architecture(to, scheme.upwinding))

@inline U_dot_∇u(i, j, k, grid, scheme::VectorInvariant, U) = horizontal_advection_U(i, j, k, grid, scheme, U.u, U.v) +
                                                                vertical_advection_U(i, j, k, grid, scheme, U) +
                                                                    bernoulli_head_U(i, j, k, grid, scheme, U.u, U.v)

@inline U_dot_∇v(i, j, k, grid, scheme::VectorInvariant, U) = horizontal_advection_V(i, j, k, grid, scheme, U.u, U.v) +
                                                                vertical_advection_V(i, j, k, grid, scheme, U) +
                                                                    bernoulli_head_V(i, j, k, grid, scheme, U.u, U.v)

# Extend interpolate functions for VectorInvariant to allow MultiDimensional reconstruction
for bias in (:_biased, :_symmetric)
    for (dir1, dir2) in zip((:xᶠᵃᵃ, :xᶜᵃᵃ, :yᵃᶠᵃ, :yᵃᶜᵃ), (:y, :y, :x, :x))
        interp_func = Symbol(bias, :_interpolate_, dir1)
        multidim_interp = Symbol(:_multi_dimensional_reconstruction_, dir2)

        @eval begin
            @inline $interp_func(i, j, k, grid, ::VectorInvariant, interp_scheme::AbstractAdvectionScheme, args...) =
                        $interp_func(i, j, k, grid, interp_scheme, args...)

            @inline $interp_func(i, j, k, grid, ::MultiDimensionalVectorInvariant, interp_scheme::AbstractAdvectionScheme, args...) =
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

@inline bernoulli_head_U(i, j, k, grid, ::VectorInvariantKEGradientEnergyConserving, u, v) = ∂xᶠᶜᶜ(i, j, k, grid, Khᶜᶜᶜ, u, v)
@inline bernoulli_head_V(i, j, k, grid, ::VectorInvariantKEGradientEnergyConserving, u, v) = ∂yᶜᶠᶜ(i, j, k, grid, Khᶜᶜᶜ, u, v)

#####
##### Conservative vertical advection
##### Follows https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#vector-invariant-momentum-equations
#####

@inbounds ζ₂wᶠᶜᶠ(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶠᶜᶠ(i, j, k, grid, u)
@inbounds ζ₁wᶜᶠᶠ(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_qᶜᶜᶠ, w) * ∂zᶜᶠᶠ(i, j, k, grid, v)

@inline vertical_advection_U(i, j, k, grid, ::VectorInvariantVerticalEnergyConserving, U) = ℑzᵃᵃᶜ(i, j, k, grid, ζ₂wᶠᶜᶠ, U.u, U.w) * Az⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline vertical_advection_V(i, j, k, grid, ::VectorInvariantVerticalEnergyConserving, U) = ℑzᵃᵃᶜ(i, j, k, grid, ζ₁wᶜᶠᶠ, U.v, U.w) * Az⁻¹ᶜᶠᶜ(i, j, k, grid)

#####
##### Upwinding vertical advection (2. and 3.)
#####

@inline function vertical_advection_U(i, j, k, grid, scheme::VectorInvariant, U)

    Φᵟ = upwinded_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme, U.u, U.v)
    𝒜ᶻ = δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, scheme.vertical_advection_scheme, U.w, U.u)

    return 1/Vᶠᶜᶜ(i, j, k, grid) * (Φᵟ + 𝒜ᶻ)
end

@inline function vertical_advection_V(i, j, k, grid, scheme::VectorInvariant, U)

    Φᵟ = upwinded_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme, U.u, U.v)
    𝒜ᶻ = δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, scheme.vertical_advection_scheme, U.w, U.v)

    return 1/Vᶜᶠᶜ(i, j, k, grid) * (Φᵟ + 𝒜ᶻ)
end

#####
##### Horizontal advection 4 formulations:
#####  1. Energy conservative
#####  2. Enstrophy conservative
#####  3. Dimension-By-Dimension Vorticity upwinding
#####  4. Two-Dimensional (x and y) Vorticity upwinding
#####

#####
##### Conserving schemes (1. and 2.)
##### Follows https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#vector-invariant-momentum-equations
#####

@inline ζ_ℑx_vᶠᶠᵃ(i, j, k, grid, u, v) = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)
@inline ζ_ℑy_uᶠᶠᵃ(i, j, k, grid, u, v) = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)

@inline horizontal_advection_U(i, j, k, grid, ::VectorInvariantEnergyConserving, u, v) = - ℑyᵃᶜᵃ(i, j, k, grid, ζ_ℑx_vᶠᶠᵃ, u, v) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline horizontal_advection_V(i, j, k, grid, ::VectorInvariantEnergyConserving, u, v) = + ℑxᶜᵃᵃ(i, j, k, grid, ζ_ℑy_uᶠᶠᵃ, u, v) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)

@inline horizontal_advection_U(i, j, k, grid, ::VectorInvariantEnstrophyConserving, u, v) = - ℑyᵃᶜᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v) * ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
@inline horizontal_advection_V(i, j, k, grid, ::VectorInvariantEnstrophyConserving, u, v) = + ℑxᶜᵃᵃ(i, j, k, grid, ζ₃ᶠᶠᶜ, u, v) * ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)

#####
##### Upwinding schemes (3. and 4.)
#####

@inline function horizontal_advection_U(i, j, k, grid, scheme::VectorInvariantUpwindVorticity, u, v)

    Sζ = scheme.vorticity_stencil

    @inbounds v̂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ζᴿ = _biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vorticity_scheme, bias(v̂), ζ₃ᶠᶠᶜ, Sζ, u, v)

    return - v̂ * ζᴿ
end

@inline function horizontal_advection_V(i, j, k, grid, scheme::VectorInvariantUpwindVorticity, u, v)

    Sζ = scheme.vorticity_stencil

    @inbounds û = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    ζᴿ = _biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vorticity_scheme, bias(û), ζ₃ᶠᶠᶜ, Sζ, u, v)

    return + û * ζᴿ
end

#####
##### Fallback to flux form advection
#####
##### Curvature metric corrections are now handled separately by the functions in
##### curvature_metric_terms.jl (U_dot_∇u_hydrostatic_metric, U_dot_∇u_metric, etc.).
#####

@inline U_dot_∇u(i, j, k, grid, advection::AbstractAdvectionScheme, U) = div_𝐯u(i, j, k, grid, advection, U, U.u)
@inline U_dot_∇v(i, j, k, grid, advection::AbstractAdvectionScheme, U) = div_𝐯v(i, j, k, grid, advection, U, U.v)

#####
##### No advection
#####

@inline U_dot_∇u(i, j, k, grid::AbstractGrid{FT}, scheme::Nothing, U) where FT = zero(FT)
@inline U_dot_∇v(i, j, k, grid::AbstractGrid{FT}, scheme::Nothing, U) where FT = zero(FT)

const AS = AbstractSmoothnessStencil

# To adapt passing smoothness stencils to upwind biased schemes and centered schemes (not WENO)
for b in advection_buffers, FT in fully_supported_float_types
    @eval begin
        @inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, s::Centered{$b, $FT}, f::Callable, ::AS, args...) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, s, f, args...)
        @inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, s::Centered{$b, $FT}, f::Callable, ::AS, args...) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, s, f, args...)
        @inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, s::Centered{$b, $FT}, f::Callable, ::AS, args...) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, s, f, args...)

        @inline biased_interpolate_xᶠᵃᵃ(i, j, k, grid, s::UpwindBiased{$b, $FT}, bias, f::Callable, ::AS, args...) = biased_interpolate_xᶠᵃᵃ(i, j, k, grid, s, bias, f, args...)
        @inline biased_interpolate_yᵃᶠᵃ(i, j, k, grid, s::UpwindBiased{$b, $FT}, bias, f::Callable, ::AS, args...) = biased_interpolate_yᵃᶠᵃ(i, j, k, grid, s, bias, f, args...)
        @inline biased_interpolate_zᵃᵃᶠ(i, j, k, grid, s::UpwindBiased{$b, $FT}, bias, f::Callable, ::AS, args...) = biased_interpolate_zᵃᵃᶠ(i, j, k, grid, s, bias, f, args...)
    end
end
