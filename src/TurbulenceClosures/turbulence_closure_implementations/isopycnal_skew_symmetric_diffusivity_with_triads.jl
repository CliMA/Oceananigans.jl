using Oceananigans.Operators: active_weighted_ℑxzᶜᶜᶜ, active_weighted_ℑyzᶜᶜᶜ

struct TriadIsopycnalSkewSymmetricDiffusivity{TD, K, S, M, L, N} <: AbstractTurbulenceClosure{TD, N}
    κ_skew :: K
    κ_symmetric :: S
    isopycnal_tensor :: M
    slope_limiter :: L

    function TriadIsopycnalSkewSymmetricDiffusivity{TD, N}(κ_skew :: K,
                                                           κ_symmetric :: S,
                                                           isopycnal_tensor :: I,
                                                           slope_limiter :: L) where {TD, K, S, I, L, N}

        return new{TD, K, S, I, L, N}(κ_skew, κ_symmetric, isopycnal_tensor, slope_limiter)
    end
end

const TISSD{TD} = TriadIsopycnalSkewSymmetricDiffusivity{TD} where TD
const TISSDVector{TD} = AbstractVector{<:TISSD{TD}} where TD
const FlavorOfTISSD{TD} = Union{TISSD{TD}, TISSDVector{TD}} where TD

"""
    TriadIsopycnalSkewSymmetricDiffusivity([time_disc=VerticallyImplicitTimeDiscretization(), FT=Float64;]
                                           κ_skew = 0,
                                           κ_symmetric = 0,
                                           isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                           slope_limiter = FluxTapering(1e-2),
                                           required_halo_size::Int = 1)

Return parameters for an isopycnal skew-symmetric tracer diffusivity with skew diffusivity
`κ_skew` and symmetric diffusivity `κ_symmetric` that uses an `isopycnal_tensor` model for
for calculating the isopycnal slopes, and (optionally) applying a `slope_limiter` to the
calculated isopycnal slope values.

Both `κ_skew` and `κ_symmetric` may be constants, arrays, fields, or functions of `(x, y, z, t)`.

The formulation follows Griffies et al. (1998)

References
==========
* Griffies, S. M., A. Gnanadesikan, R. C. Pacanowski, V. D. Larichev, J. K. Dukowicz, and R. D. Smith (1998) Isoneutral diffusion in a z-coordinate ocean model. _J. Phys. Oceanogr._, **28**, 805–830, doi:10.1175/1520-0485(1998)028<0805:IDIAZC>2.0.CO;2
"""
function TriadIsopycnalSkewSymmetricDiffusivity(time_disc=ExplicitTimeDiscretization(), FT=Float64;
                                                κ_skew = 0,
                                                κ_symmetric = 0,
                                                isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                                slope_limiter = FluxTapering(1e-2),
                                                required_halo_size::Int = 1)

    isopycnal_tensor isa SmallSlopeIsopycnalTensor ||
        error("Only isopycnal_tensor=SmallSlopeIsopycnalTensor() is currently supported.")

    TD = typeof(time_disc)

    return TriadIsopycnalSkewSymmetricDiffusivity{TD, required_halo_size}(convert_diffusivity(FT, κ_skew),
                                                                          convert_diffusivity(FT, κ_symmetric),
                                                                          isopycnal_tensor,
                                                                          slope_limiter)
end

TriadIsopycnalSkewSymmetricDiffusivity(FT::DataType; kw...) =
    TriadIsopycnalSkewSymmetricDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

Utils.with_tracers(tracers, closure::TISSD{TD, N}) where {TD, N} =
    TriadIsopycnalSkewSymmetricDiffusivity{TD, N}(closure.κ_skew, closure.κ_symmetric, closure.isopycnal_tensor, closure.slope_limiter)

# For ensembles of closures
function Utils.with_tracers(tracers, closure_vector::TISSDVector)
    arch = architecture(closure_vector)

    _closure_vector = arch isa Architectures.GPU ? Vector(closure_vector) : closure_vector

    Ex = length(_closure_vector)
    vec = [with_tracers(tracers, _closure_vector[i]) for i=1:Ex]

    return on_architecture(arch, vec)
end

# Note: computing diffusivities at cell centers for now.
function DiffusivityFields(grid, tracer_names, bcs, ::FlavorOfTISSD{TD}) where TD
    if TD() isa VerticallyImplicitTimeDiscretization
        # Precompute the _tapered_ 33 component of the isopycnal rotation tensor
        K = (; ϵκR₃₃ = ZFaceField(grid))
    else
        return nothing
    end

    return K
end

# Build closure fields for model initialization
build_closure_fields(grid, clock, tracer_names, bcs, closure::FlavorOfTISSD) =
    DiffusivityFields(grid, tracer_names, bcs, closure)

function compute_closure_fields!(closure_fields, closure::FlavorOfTISSD{TD}, model; parameters = :xyz) where TD

    arch = model.architecture
    grid = model.grid
    clock = model.clock
    tracers = buoyancy_tracers(model)
    buoyancy = buoyancy_force(model)

    if TD() isa VerticallyImplicitTimeDiscretization
        launch!(arch, grid, parameters,
                triad_compute_tapered_R₃₃!,
                closure_fields, grid, closure, clock, buoyancy, tracers)
    end

    return nothing
end

@kernel function triad_compute_tapered_R₃₃!(K, grid, closure, clock, b, C)
    i, j, k, = @index(Global, NTuple)
    closure = getclosure(i, j, closure)
    κ  = closure.κ_symmetric
    sl = closure.slope_limiter
    @inbounds K.ϵκR₃₃[i, j, k] = ϵκR₃₃(i, j, k, grid, κ, clock, sl, b, C)
end

#####
##### _triads_
#####
##### There are two horizontal slopes: Sx and Sy
#####
##### Both slopes are "located" at tracer cell centers.
#####
##### The slopes are computed by a directional derivative, which lends an
##### "orientation" to the slope. For example, the x-slope `Sx` computed
##### with a "+" directional derivative in x, and a "+" directional derivative
##### in z, is
#####
##### Sx⁺⁺ᵢₖ = Δz / Δx * (bᵢ₊₁ - bᵢ) / (bₖ₊₁ - bₖ)
#####
##### The superscript codes ⁺⁺, ⁺⁻, ⁻⁺, ⁻⁻, denote the direction of the derivative
##### in (h, z).
#####
##### from https://github.com/CliMA/Oceananigans.jl/blob/glw/homogeneous-bounded/src/TurbulenceClosures/turbulence_closure_implementations/isopycnal_potential_vorticity_diffusivity.jl
#####

@inline function triad_Sx(ix, iz, j, kx, kz, grid, buoyancy, tracers)
    bx = ∂xᵣ_b(ix, j, kx, grid, buoyancy, tracers)
    bz =  ∂z_b(iz, j, kz, grid, buoyancy, tracers)
    bz = max(bz, zero(grid))
    return ifelse(bz == 0, zero(grid), - bx / bz)
end

@inline function triad_Sy(i, jy, jz, ky, kz, grid, buoyancy, tracers)
    by = ∂yᵣ_b(i, jy, ky, grid, buoyancy, tracers)
    bz =  ∂z_b(i, jz, kz, grid, buoyancy, tracers)
    bz = max(bz, zero(grid))
    return ifelse(bz == 0, zero(grid), - by / bz)
end

@inline Sx⁺⁺(i, j, k, grid, buoyancy, tracers) = triad_Sx(i+1, i, j, k, k+1, grid, buoyancy, tracers)
@inline Sx⁺⁻(i, j, k, grid, buoyancy, tracers) = triad_Sx(i+1, i, j, k, k,   grid, buoyancy, tracers)
@inline Sx⁻⁺(i, j, k, grid, buoyancy, tracers) = triad_Sx(i,   i, j, k, k+1, grid, buoyancy, tracers)
@inline Sx⁻⁻(i, j, k, grid, buoyancy, tracers) = triad_Sx(i,   i, j, k, k,   grid, buoyancy, tracers)

@inline Sy⁺⁺(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j+1, j, k, k+1, grid, buoyancy, tracers)
@inline Sy⁺⁻(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j+1, j, k, k,   grid, buoyancy, tracers)
@inline Sy⁻⁺(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j,   j, k, k+1, grid, buoyancy, tracers)
@inline Sy⁻⁻(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j,   j, k, k,   grid, buoyancy, tracers)

# We remove triads that live on a boundary (immersed or top / bottom / north / south / east / west)
@inline triad_mask_x(ix, iz, j, kx, kz, grid) =
   !peripheral_node(ix, j, kx, grid, Face(), Center(), Center()) & !peripheral_node(iz, j, kz, grid, Center(), Center(), Face())

@inline triad_mask_y(i, jy, jz, ky, kz, grid) =
   !peripheral_node(i, jy, ky, grid, Center(), Face(), Center()) & !peripheral_node(i, jz, kz, grid, Center(), Center(), Face())

@inline ϵκx⁺⁺(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_x(i+1, i, j, k, k+1, grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock, C) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)
@inline ϵκx⁺⁻(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_x(i+1, i, j, k, k,   grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock, C) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)
@inline ϵκx⁻⁺(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_x(i,   i, j, k, k+1, grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock, C) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)
@inline ϵκx⁻⁻(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_x(i,   i, j, k, k,   grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock, C) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)

@inline ϵκy⁺⁺(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_y(i, j+1, j, k, k+1, grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock, C) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)
@inline ϵκy⁺⁻(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_y(i, j+1, j, k, k,   grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock, C) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)
@inline ϵκy⁻⁺(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_y(i, j,   j, k, k+1, grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock, C) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)
@inline ϵκy⁻⁻(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_y(i, j,   j, k, k,   grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock, C) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)

# Triad diagram key
# =================
#
#   * ┗ : Sx⁺⁺ / Sy⁺⁺
#   * ┛ : Sx⁻⁺ / Sy⁻⁺
#   * ┓ : Sx⁻⁻ / Sy⁻⁻
#   * ┏ : Sx⁺⁻ / Sy⁺⁻
#

# defined at fcc
@inline function diffusive_flux_x(i, j, k, grid, closure::FlavorOfTISSD, K, ::Val{id},
                                  c, clock, C, b) where id

    closure = getclosure(i, j, closure)
    κ  = closure.κ_symmetric
    sl = closure.slope_limiter
    loc = (Center(), Center(), Center())

    ϵκ⁺⁺ = ϵκx⁺⁺(i-1, j, k, grid, loc, κ, clock, sl, b, C)
    ϵκ⁺⁻ = ϵκx⁺⁻(i-1, j, k, grid, loc, κ, clock, sl, b, C)
    ϵκ⁻⁺ = ϵκx⁻⁺(i,   j, k, grid, loc, κ, clock, sl, b, C)
    ϵκ⁻⁻ = ϵκx⁻⁻(i,   j, k, grid, loc, κ, clock, sl, b, C)

    # Small slope approximation
    ∂x_c = ∂xᵣᶠᶜᶜ(i, j, k, grid, c)

    #       i-1     i
    # k+1  -------------
    #           |      |
    #       ┏┗  ∘  ┛┓  | k
    #           |      |
    # k   ------|------|

    Fx = (ϵκ⁺⁺ * (∂x_c + Sx⁺⁺(i-1, j, k, grid, b, C) * ∂zᶜᶜᶠ(i-1, j, k+1, grid, c)) +
          ϵκ⁺⁻ * (∂x_c + Sx⁺⁻(i-1, j, k, grid, b, C) * ∂zᶜᶜᶠ(i-1, j, k,   grid, c)) +
          ϵκ⁻⁺ * (∂x_c + Sx⁻⁺(i,   j, k, grid, b, C) * ∂zᶜᶜᶠ(i,   j, k+1, grid, c)) +
          ϵκ⁻⁻ * (∂x_c + Sx⁻⁻(i,   j, k, grid, b, C) * ∂zᶜᶜᶠ(i,   j, k,   grid, c))) / 4

    return - Fx
end

# defined at cfc
@inline function diffusive_flux_y(i, j, k, grid, closure::FlavorOfTISSD, K, ::Val{id},
                                  c, clock, C, b) where id

    closure = getclosure(i, j, closure)
    κ  = closure.κ_symmetric
    sl = closure.slope_limiter
    loc = (Center(), Center(), Center())

    ∂y_c = ∂yᵣᶜᶠᶜ(i, j, k, grid, c)

    ϵκ⁺⁺ = ϵκy⁺⁺(i, j-1, k, grid, loc, κ, clock, sl, b, C)
    ϵκ⁺⁻ = ϵκy⁺⁻(i, j-1, k, grid, loc, κ, clock, sl, b, C)
    ϵκ⁻⁺ = ϵκy⁻⁺(i, j,   k, grid, loc, κ, clock, sl, b, C)
    ϵκ⁻⁻ = ϵκy⁻⁻(i, j,   k, grid, loc, κ, clock, sl, b, C)

    Fy = (ϵκ⁺⁺ * (∂y_c + Sy⁺⁺(i, j-1, k, grid, b, C) * ∂zᶜᶜᶠ(i, j-1, k+1, grid, c)) +
          ϵκ⁺⁻ * (∂y_c + Sy⁺⁻(i, j-1, k, grid, b, C) * ∂zᶜᶜᶠ(i, j-1, k,   grid, c)) +
          ϵκ⁻⁺ * (∂y_c + Sy⁻⁺(i, j,   k, grid, b, C) * ∂zᶜᶜᶠ(i, j,   k+1, grid, c)) +
          ϵκ⁻⁻ * (∂y_c + Sy⁻⁻(i, j,   k, grid, b, C) * ∂zᶜᶜᶠ(i, j,   k,   grid, c))) / 4

    return - Fy
end

# defined at ccf
@inline function diffusive_flux_z(i, j, k, grid, closure::FlavorOfTISSD{TD}, K, ::Val{id},
                                  c, clock, C, b) where {TD, id}

    closure = getclosure(i, j, closure)
    κ  = closure.κ_symmetric
    sl = closure.slope_limiter

    loc = (Center(), Center(), Center())

    ϵκˣ⁻⁻ = ϵκx⁻⁻(i, j, k,   grid, loc, κ, clock, sl, b, C)
    ϵκˣ⁺⁻ = ϵκx⁺⁻(i, j, k,   grid, loc, κ, clock, sl, b, C)
    ϵκˣ⁻⁺ = ϵκx⁻⁺(i, j, k-1, grid, loc, κ, clock, sl, b, C)
    ϵκˣ⁺⁺ = ϵκx⁺⁺(i, j, k-1, grid, loc, κ, clock, sl, b, C)

    ϵκʸ⁻⁻ = ϵκy⁻⁻(i, j, k,   grid, loc, κ, clock, sl, b, C)
    ϵκʸ⁺⁻ = ϵκy⁺⁻(i, j, k,   grid, loc, κ, clock, sl, b, C)
    ϵκʸ⁻⁺ = ϵκy⁻⁺(i, j, k-1, grid, loc, κ, clock, sl, b, C)
    ϵκʸ⁺⁺ = ϵκy⁺⁺(i, j, k-1, grid, loc, κ, clock, sl, b, C)

    # Triad diagram:
    #
    #   i-1    i    i+1
    # -------------------
    # |     |     |     |
    # |     | ┓ ┏ |  k  |
    # |     |     |     |
    # -  k  -- ∘ --     -
    # |     |     |     |
    # |     | ┛ ┗ | k-1 |
    # |     |     |     |
    # --------------------

    κR₃₁_∂x_c = (ϵκˣ⁻⁻ * Sx⁻⁻(i, j, k,   grid, b, C) * ∂xᵣᶠᶜᶜ(i,   j, k,   grid, c) +
                 ϵκˣ⁺⁻ * Sx⁺⁻(i, j, k,   grid, b, C) * ∂xᵣᶠᶜᶜ(i+1, j, k,   grid, c) +
                 ϵκˣ⁻⁺ * Sx⁻⁺(i, j, k-1, grid, b, C) * ∂xᵣᶠᶜᶜ(i,   j, k-1, grid, c) +
                 ϵκˣ⁺⁺ * Sx⁺⁺(i, j, k-1, grid, b, C) * ∂xᵣᶠᶜᶜ(i+1, j, k-1, grid, c)) / 4

    κR₃₂_∂y_c = (ϵκʸ⁻⁻ * Sy⁻⁻(i, j, k,   grid, b, C) * ∂yᵣᶜᶠᶜ(i, j,   k,   grid, c) +
                 ϵκʸ⁺⁻ * Sy⁺⁻(i, j, k,   grid, b, C) * ∂yᵣᶜᶠᶜ(i, j+1, k,   grid, c) +
                 ϵκʸ⁻⁺ * Sy⁻⁺(i, j, k-1, grid, b, C) * ∂yᵣᶜᶠᶜ(i, j,   k-1, grid, c) +
                 ϵκʸ⁺⁺ * Sy⁺⁺(i, j, k-1, grid, b, C) * ∂yᵣᶜᶠᶜ(i, j+1, k-1, grid, c)) / 4

    κϵ_R₃₃_∂z_c = explicit_R₃₃_∂z_c(i, j, k, grid, TD(), clock, c, closure, b, C)

    return - κR₃₁_∂x_c - κR₃₂_∂y_c - κϵ_R₃₃_∂z_c
end

@inline function ϵκR₃₃(i, j, k, grid, κ, clock, sl, b, C)
    loc = (Center(), Center(), Center())

    ϵκˣ⁻⁻ = ϵκx⁻⁻(i, j, k,   grid, loc, κ, clock, sl, b, C)
    ϵκˣ⁺⁻ = ϵκx⁺⁻(i, j, k,   grid, loc, κ, clock, sl, b, C)
    ϵκˣ⁻⁺ = ϵκx⁻⁺(i, j, k-1, grid, loc, κ, clock, sl, b, C)
    ϵκˣ⁺⁺ = ϵκx⁺⁺(i, j, k-1, grid, loc, κ, clock, sl, b, C)

    ϵκʸ⁻⁻ = ϵκy⁻⁻(i, j, k,   grid, loc, κ, clock, sl, b, C)
    ϵκʸ⁺⁻ = ϵκy⁺⁻(i, j, k,   grid, loc, κ, clock, sl, b, C)
    ϵκʸ⁻⁺ = ϵκy⁻⁺(i, j, k-1, grid, loc, κ, clock, sl, b, C)
    ϵκʸ⁺⁺ = ϵκy⁺⁺(i, j, k-1, grid, loc, κ, clock, sl, b, C)

    ϵκR₃₃ = (ϵκˣ⁻⁻ * Sx⁻⁻(i, j, k,   grid, b, C)^2 + ϵκʸ⁻⁻ * Sy⁻⁻(i, j, k,   grid, b, C)^2 +
             ϵκˣ⁺⁻ * Sx⁺⁻(i, j, k,   grid, b, C)^2 + ϵκʸ⁺⁻ * Sy⁺⁻(i, j, k,   grid, b, C)^2 +
             ϵκˣ⁻⁺ * Sx⁻⁺(i, j, k-1, grid, b, C)^2 + ϵκʸ⁻⁺ * Sy⁻⁺(i, j, k-1, grid, b, C)^2 +
             ϵκˣ⁺⁺ * Sx⁺⁺(i, j, k-1, grid, b, C)^2 + ϵκʸ⁺⁺ * Sy⁺⁺(i, j, k-1, grid, b, C)^2) / 4

    return ϵκR₃₃
end

@inline function explicit_R₃₃_∂z_c(i, j, k, grid, ::ExplicitTimeDiscretization, clock, c, closure, b, C)
    κ  = closure.κ_symmetric
    sl = closure.slope_limiter
    return ϵκR₃₃(i, j, k, grid, κ, clock, sl, b, C) * ∂zᶜᶜᶠ(i, j, k, grid, c)
end

@inline explicit_R₃₃_∂z_c(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, clock, c, closure, b, C) = zero(grid)

@inline κzᶜᶜᶠ(i, j, k, grid, closure::FlavorOfTISSD, K, ::Val{id}, clock) where id = @inbounds K.ϵκR₃₃[i, j, k]

@inline viscous_flux_ux(i, j, k, grid, closure::Union{TISSD, TISSDVector}, args...) = zero(grid)
@inline viscous_flux_uy(i, j, k, grid, closure::Union{TISSD, TISSDVector}, args...) = zero(grid)
@inline viscous_flux_uz(i, j, k, grid, closure::Union{TISSD, TISSDVector}, args...) = zero(grid)

@inline viscous_flux_vx(i, j, k, grid, closure::Union{TISSD, TISSDVector}, args...) = zero(grid)
@inline viscous_flux_vy(i, j, k, grid, closure::Union{TISSD, TISSDVector}, args...) = zero(grid)
@inline viscous_flux_vz(i, j, k, grid, closure::Union{TISSD, TISSDVector}, args...) = zero(grid)

@inline viscous_flux_wx(i, j, k, grid, closure::Union{TISSD, TISSDVector}, args...) = zero(grid)
@inline viscous_flux_wy(i, j, k, grid, closure::Union{TISSD, TISSDVector}, args...) = zero(grid)
@inline viscous_flux_wz(i, j, k, grid, closure::Union{TISSD, TISSDVector}, args...) = zero(grid)

#####
##### Show
#####

Base.summary(closure::TISSD) = string("TriadIsopycnalSkewSymmetricDiffusivity",
                                     "(κ_skew=",
                                     prettysummary(closure.κ_skew),
                                     ", κ_symmetric=", prettysummary(closure.κ_symmetric), ")")

Base.show(io::IO, closure::TISSD) =
    print(io, "TriadIsopycnalSkewSymmetricDiffusivity: " *
              "(κ_symmetric=$(closure.κ_symmetric), κ_skew=$(closure.κ_skew), " *
              "(isopycnal_tensor=$(closure.isopycnal_tensor), slope_limiter=$(closure.slope_limiter))")

@inline not_peripheral_node(args...) = !peripheral_node(args...)

# the `tapering_factor` function as well as the slope function `Sxᶠᶜᶠ` and `Syᶜᶠᶠ`
# are defined in the `advective_skew_diffusion.jl` file
@inline function tapering_factorᶜᶜᶜ(i, j, k, grid, slope_limiter, buoyancy, tracers)
    Sx = active_weighted_ℑxzᶜᶜᶜ(i, j, k, grid, Sxᶠᶜᶠ, buoyancy, tracers)
    Sy = active_weighted_ℑyzᶜᶜᶜ(i, j, k, grid, Syᶜᶠᶠ, buoyancy, tracers)
    return tapering_factor(Sx, Sy, slope_limiter)
end
