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
const issd_coefficient_loc = (Center(), Center(), Center())

"""
    TriadIsopycnalSkewSymmetricDiffusivity([time_disc=VerticallyImplicitTimeDiscretization(), FT=Float64;]
                                           κ_skew = 0,
                                           κ_symmetric = 0,
                                           isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                           slope_limiter = FluxTapering(1e-2))

Return parameters for an isopycnal skew-symmetric tracer diffusivity with skew diffusivity
`κ_skew` and symmetric diffusivity `κ_symmetric` that uses an `isopycnal_tensor` model for
for calculating the isopycnal slopes, and (optionally) applying a `slope_limiter` to the
calculated isopycnal slope values.
    
Both `κ_skew` and `κ_symmetric` may be constants, arrays, fields, or functions of `(x, y, z, t)`.
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

function with_tracers(tracers, closure::TISSD{TD, N}) where {TD, N}
    κ_skew = !isa(closure.κ_skew, NamedTuple) ? closure.κ_skew : tracer_diffusivities(tracers, closure.κ_skew)
    κ_symmetric = !isa(closure.κ_symmetric, NamedTuple) ? closure.κ_symmetric : tracer_diffusivities(tracers, closure.κ_symmetric)
    return TriadIsopycnalSkewSymmetricDiffusivity{TD, N}(κ_skew, κ_symmetric, closure.isopycnal_tensor, closure.slope_limiter)
end

# For ensembles of closures
function with_tracers(tracers, closure_vector::TISSDVector)
    arch = architecture(closure_vector)

    if arch isa Architectures.GPU
        closure_vector = Vector(closure_vector)
    end

    Ex = length(closure_vector)
    closure_vector = [with_tracers(tracers, closure_vector[i]) for i=1:Ex]

    return on_architecture(arch, closure_vector)
end

# Note: computing diffusivities at cell centers for now.
function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfTISSD{TD}) where TD
    if TD() isa VerticallyImplicitTimeDiscretization
        # Precompute the _tapered_ 33 component of the isopycnal rotation tensor
        return (; ϵ_R₃₃ = Field((Center, Center, Face), grid))
    else
        return nothing
    end
end

function compute_diffusivities!(diffusivities, closure::FlavorOfTISSD{TD}, model; parameters = :xyz) where TD

    arch = model.architecture
    grid = model.grid
    tracers = model.tracers
    buoyancy = model.buoyancy

    if TD() isa VerticallyImplicitTimeDiscretization
        launch!(arch, grid, parameters,
                triad_compute_tapered_R₃₃!,
                diffusivities.ϵ_R₃₃, grid, closure, tracers, buoyancy)
    end

    return nothing
end

@kernel function triad_compute_tapered_R₃₃!(ϵ_R₃₃, grid, closure, tracers, buoyancy) 
    i, j, k, = @index(Global, NTuple)
    closure = getclosure(i, j, closure)
    R₃₃ = isopycnal_rotation_tensor_zz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    ϵ = tapering_factorᶜᶜᶠ(i, j, k, grid, closure, tracers, buoyancy)
    @inbounds ϵ_R₃₃[i, j, k] = ϵ * R₃₃
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

@inline function triad_Sx(ix, jx, kx, iz, jz, kz, grid, buoyancy, tracers)
    bx = ∂x_b(ix, jx, kx, grid, buoyancy, tracers)
    bz = ∂z_b(iz, jz, kz, grid, buoyancy, tracers)
    bz = max(bz, zero(grid))
    return ifelse(bz == 0, zero(grid), - bx / bz)
end

@inline function triad_Sy(iy, jy, ky, iz, jz, kz, grid, buoyancy, tracers)
    by = ∂y_b(iy, jy, ky, grid, buoyancy, tracers)
    bz = ∂z_b(iz, jz, kz, grid, buoyancy, tracers)
    bz = max(bz, zero(grid))
    return ifelse(bz == 0, zero(grid), - by / bz)
end

@inline Sx⁺⁺(i, j, k, grid, buoyancy, tracers) = triad_Sx(i+1, j, k, i, j, k+1, grid, buoyancy, tracers)
@inline Sx⁺⁻(i, j, k, grid, buoyancy, tracers) = triad_Sx(i+1, j, k, i, j, k,   grid, buoyancy, tracers)
@inline Sx⁻⁺(i, j, k, grid, buoyancy, tracers) = triad_Sx(i,   j, k, i, j, k+1, grid, buoyancy, tracers)
@inline Sx⁻⁻(i, j, k, grid, buoyancy, tracers) = triad_Sx(i,   j, k, i, j, k,   grid, buoyancy, tracers)

@inline Sy⁺⁺(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j+1, k, i, j, k+1, grid, buoyancy, tracers)
@inline Sy⁺⁻(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j+1, k, i, j, k,   grid, buoyancy, tracers)
@inline Sy⁻⁺(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j,   k, i, j, k+1, grid, buoyancy, tracers)
@inline Sy⁻⁻(i, j, k, grid, buoyancy, tracers) = triad_Sy(i, j,   k, i, j, k,   grid, buoyancy, tracers)

# Tensor components
@inline Sxᶠᶜᶜ(i, j, k, grid, args...) = (Sx⁻⁺(i,   j, k, grid, args...) + Sx⁻⁻(i,   j, k, grid, args...) +    # west cell, z-average
                                         Sx⁻⁺(i-1, j, k, grid, args...) + Sx⁻⁻(i-1, j, k, grid, args...)) / 4 # east cell, z-average

@inline Syᶜᶠᶜ(i, j, k, grid, args...) = (Sy⁻⁺(i, j,   k, grid, args...) + Sy⁻⁻(i, j,   k, grid, args...) +    # south cell, z-average
                                         Sy⁻⁺(i, j-1, k, grid, args...) + Sy⁻⁻(i, j-1, k, grid, args...)) / 4 # north cell, z-average

@inline Syᶜᶜᶠ(i, j, k, grid, args...) = (Sy⁺⁻(i, j, k,   grid, args...) + Sy⁻⁻(i, j, k,   grid, args...) +    # top cell, y-average
                                         Sy⁺⁺(i, j, k-1, grid, args...) + Sy⁻⁺(i, j, k-1, grid, args...)) / 4 # bottom cell, y-average

@inline Sxᶜᶜᶠ(i, j, k, grid, args...) = (Sx⁺⁻(i, j, k,   grid, args...) + Sx⁻⁻(i, j, k,   grid, args...) +    # top cell, x-average
                                         Sx⁺⁺(i, j, k-1, grid, args...) + Sx⁻⁺(i, j, k-1, grid, args...)) / 4 # bottom cell, x-average

@inline Syᶜᶜᶠ(i, j, k, grid, args...) = (Sy⁺⁻(i, j, k,   grid, args...) + Sy⁻⁻(i, j, k,   grid, args...) +    # top cell, y-average
                                         Sy⁺⁺(i, j, k-1, grid, args...) + Sy⁻⁺(i, j, k-1, grid, args...)) / 4 # bottom cell, y-average




struct FluxTapering{FT}
    max_slope :: FT
end

"""
    taper_factor(i, j, k, grid, closure, tracers, buoyancy) 

Return the tapering factor `min(1, Sₘₐₓ² / slope²)`, where `slope² = slope_x² + slope_y²`
that multiplies all components of the isopycnal slope tensor. The tapering factor is calculated on all the
faces involved in the isopycnal slope tensor calculation. The minimum value of tapering is selected.

References
==========
R. Gerdes, C. Koberle, and J. Willebrand. (1991), "The influence of numerical advection schemes
    on the results of ocean general circulation models", Clim. Dynamics, 5 (4), 211–226.
"""
@inline function tapering_factor(i, j, k, grid, closure, tracers, buoyancy)
    ϵᶠᶜᶜ = tapering_factorᶠᶜᶜ(i, j, k, grid, closure, tracers, buoyancy)
    ϵᶜᶠᶜ = tapering_factorᶜᶠᶜ(i, j, k, grid, closure, tracers, buoyancy)
    ϵᶜᶜᶠ = tapering_factorᶜᶜᶠ(i, j, k, grid, closure, tracers, buoyancy)
    return min(ϵᶠᶜᶜ, ϵᶜᶠᶜ, ϵᶜᶜᶠ)
end

@inline function tapering_factorᶠᶜᶜ(i, j, k, grid, closure, tracers, buoyancy)
    by = ℑxyᶠᶜᵃ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ℑxzᶠᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    bx = ∂x_b(i, j, k, grid, buoyancy, tracers)
    return calc_tapering(bx, by, bz, grid, closure.isopycnal_tensor, closure.slope_limiter)
end

@inline function tapering_factorᶜᶠᶜ(i, j, k, grid, closure, tracers, buoyancy)
    bx = ℑxyᶜᶠᵃ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    bz = ℑyzᵃᶠᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    by = ∂y_b(i, j, k, grid, buoyancy, tracers)
    return calc_tapering(bx, by, bz, grid, closure.isopycnal_tensor, closure.slope_limiter)
end

@inline function tapering_factorᶜᶜᶠ(i, j, k, grid, closure, tracers, buoyancy)
    bx = ℑxzᶜᵃᶠ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    by = ℑyzᵃᶜᶠ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return calc_tapering(bx, by, bz, grid, closure.isopycnal_tensor, closure.slope_limiter)
end

@inline function calc_tapering(bx, by, bz, grid, slope_model, slope_limiter)
    
    bz = max(bz, slope_model.minimum_bz)
    
    slope_x = - bx / bz
    slope_y = - by / bz
   
    # in case of a stable buoyancy gradient (bz > 0), the slope is set to zero
    slope² = ifelse(bz <= 0, zero(grid), slope_x^2 + slope_y^2) 

    return min(one(grid), slope_limiter.max_slope^2 / slope²)
end

# Diffusive fluxes

@inline get_tracer_κ(κ::NamedTuple, tracer_index) = @inbounds κ[tracer_index]
@inline get_tracer_κ(κ, tracer_index) = κ


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

    κ = get_tracer_κ(closure.κ_skew, id)
    κ = κᶠᶜᶜ(i, j, k, grid, issd_coefficient_loc, κ, clock)

    # Small slope approximation
    R₁₁_∂x_c = ∂xᶠᶜᶜ(i, j, k, grid, c)
    R₁₂_∂y_c = zero(grid)

    #       i-1     i 
    # k+1  -------------
    #           |      |
    #       ┏┗  ∘  ┛┓  | k
    #           |      |
    # k   ------|------|

    R₁₃_∂z_c = (Sx⁺⁺(i-1, j, k, grid, b, C) * ∂zᶜᶜᶠ(i-1, j, k+1, grid, c) +
                Sx⁺⁻(i-1, j, k, grid, b, C) * ∂zᶜᶜᶠ(i-1, j, k,   grid, c) +
                Sx⁻⁺(i,   j, k, grid, b, C) * ∂zᶜᶜᶠ(i,   j, k+1, grid, c) +
                Sx⁻⁻(i,   j, k, grid, b, C) * ∂zᶜᶜᶠ(i,   j, k,   grid, c)) / 4
    
    return - κ * (R₁₁_∂x_c + R₁₂_∂y_c + R₁₃_∂z_c)
end

# defined at cfc
@inline function diffusive_flux_y(i, j, k, grid, closure::FlavorOfTISSD, K, ::Val{id},
                                  c, clock, C, b) where id

    closure = getclosure(i, j, closure)

    κ = get_tracer_κ(closure.κ_skew, id)
    κ = κᶜᶠᶜ(i, j, k, grid, issd_coefficient_loc, κ, clock)

    # Small slope approximation
    R₂₁_∂x_c = zero(grid)
    R₂₂_∂y_c = ∂yᶜᶠᶜ(i, j, k, grid, c)

    R₂₃_∂z_c = (Sy⁺⁺(i, j-1, k, grid, b, C) * ∂zᶜᶜᶠ(i, j-1, k+1, grid, c) +
                Sy⁺⁻(i, j-1, k, grid, b, C) * ∂zᶜᶜᶠ(i, j-1, k,   grid, c) +
                Sy⁻⁺(i, j,   k, grid, b, C) * ∂zᶜᶜᶠ(i, j,   k+1, grid, c) +
                Sy⁻⁻(i, j,   k, grid, b, C) * ∂zᶜᶜᶠ(i, j,   k,   grid, c)) / 4
    
    return - κ * (R₂₁_∂x_c + R₂₂_∂y_c + R₂₃_∂z_c)
end

# defined at ccf
@inline function diffusive_flux_z(i, j, k, grid, closure::FlavorOfTISSD{TD}, K, ::Val{id},
                                  c, clock, C, b) where {TD, id}

    closure = getclosure(i, j, closure)

    κ = get_tracer_κ(closure.κ_skew, id)
    κ = κᶜᶜᶠ(i, j, k, grid, issd_coefficient_loc, κ, clock)

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
    
    R₃₁_∂x_c = (Sx⁻⁻(i, j, k,   grid, b, C) * ∂xᶠᶜᶜ(i,   j, k,   grid, c) +
                Sx⁺⁻(i, j, k,   grid, b, C) * ∂xᶠᶜᶜ(i+1, j, k,   grid, c) +
                Sx⁻⁺(i, j, k-1, grid, b, C) * ∂xᶠᶜᶜ(i,   j, k-1, grid, c) +
                Sx⁺⁺(i, j, k-1, grid, b, C) * ∂xᶠᶜᶜ(i+1, j, k-1, grid, c)) / 4

    R₃₂_∂y_c = (Sy⁻⁻(i, j, k,   grid, b, C) * ∂yᶜᶠᶜ(i, j,   k,   grid, c) +
                Sy⁺⁻(i, j, k,   grid, b, C) * ∂yᶜᶠᶜ(i, j+1, k,   grid, c) +
                Sy⁻⁺(i, j, k-1, grid, b, C) * ∂yᶜᶠᶜ(i, j,   k-1, grid, c) +
                Sy⁺⁺(i, j, k-1, grid, b, C) * ∂yᶜᶠᶜ(i, j+1, k-1, grid, c)) / 4

    ϵ_R₃₃_∂z_c = explicit_R₃₃_∂z_c(i, j, k, grid, TD(), c, closure, b, C)

    return - κ * (R₃₁_∂x_c + R₃₂_∂y_c + ϵ_R₃₃_∂z_c)
end

@inline function explicit_R₃₃_∂z_c(i, j, k, grid, ::ExplicitTimeDiscretization, c, closure, b, C)
    ϵ = tapering_factorᶜᶜᶠ(i, j, k, grid, closure, C, b)
    ∂z_c = ∂zᶜᶜᶠ(i, j, k, grid, c)
    Sx = Sxᶜᶜᶠ(i, j, k, grid, b, C)
    Sy = Syᶜᶜᶠ(i, j, k, grid, b, C)
    R₃₃ = Sx^2 + Sy^2
    return ϵ * R₃₃ * ∂z_c
end

#=
# defined at fcc
@inline function diffusive_flux_x(i, j, k, grid,
                                  closure::Union{TISSD, TISSDVector}, diffusivity_fields, ::Val{tracer_index},
                                  c, clock, fields, buoyancy) where tracer_index

    closure = getclosure(i, j, closure)

    κ_skew = get_tracer_κ(closure.κ_skew, tracer_index)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, tracer_index)

    κ_skewᶠᶜᶜ = κᶠᶜᶜ(i, j, k, grid, issd_coefficient_loc, κ_skew, clock)
    κ_symmetricᶠᶜᶜ = κᶠᶜᶜ(i, j, k, grid, issd_coefficient_loc, κ_symmetric, clock)

    ∂x_c = ∂xᶠᶜᶜ(i, j, k, grid, c)

    # Average... of... the gradient!
    ∂y_c = ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, c)
    ∂z_c = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, c)

    R₁₁ = one(grid)
    R₁₂ = zero(grid)
    R₁₃ = isopycnal_rotation_tensor_xz_fcc(i, j, k, grid, buoyancy, fields, closure.isopycnal_tensor)
    
    ϵ = tapering_factorᶠᶜᶜ(i, j, k, grid, closure, fields, buoyancy)

    return  - ϵ * ( κ_symmetricᶠᶜᶜ * R₁₁ * ∂x_c +
                    κ_symmetricᶠᶜᶜ * R₁₂ * ∂y_c +
                   (κ_symmetricᶠᶜᶜ - κ_skewᶠᶜᶜ) * R₁₃ * ∂z_c)
end

# defined at cfc
@inline function diffusive_flux_y(i, j, k, grid,
                                  closure::Union{TISSD, TISSDVector}, diffusivity_fields, ::Val{tracer_index},
                                  c, clock, fields, buoyancy) where tracer_index

    closure = getclosure(i, j, closure)

    κ_skew = get_tracer_κ(closure.κ_skew, tracer_index)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, tracer_index)

    κ_skewᶜᶠᶜ = κᶜᶠᶜ(i, j, k, grid, issd_coefficient_loc, κ_skew, clock)
    κ_symmetricᶜᶠᶜ = κᶜᶠᶜ(i, j, k, grid, issd_coefficient_loc, κ_symmetric, clock)

    ∂y_c = ∂yᶜᶠᶜ(i, j, k, grid, c)

    # Average... of... the gradient!
    ∂x_c = ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, c)
    ∂z_c = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, c)

    R₂₁ = zero(grid)
    R₂₂ = one(grid)
    R₂₃ = isopycnal_rotation_tensor_yz_cfc(i, j, k, grid, buoyancy, fields, closure.isopycnal_tensor)

    ϵ = tapering_factorᶜᶠᶜ(i, j, k, grid, closure, fields, buoyancy)

    return - ϵ * (κ_symmetricᶜᶠᶜ * R₂₁ * ∂x_c +
                  κ_symmetricᶜᶠᶜ * R₂₂ * ∂y_c +
                 (κ_symmetricᶜᶠᶜ - κ_skewᶜᶠᶜ) * R₂₃ * ∂z_c)
end

# defined at ccf
@inline function diffusive_flux_z(i, j, k, grid,
                                  closure::FlavorOfTISSD{TD}, diffusivity_fields, ::Val{tracer_index},
                                  c, clock, fields, buoyancy) where {tracer_index, TD}

    closure = getclosure(i, j, closure)

    κ_skew = get_tracer_κ(closure.κ_skew, tracer_index)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, tracer_index)

    κ_skewᶜᶜᶠ = κᶜᶜᶠ(i, j, k, grid, issd_coefficient_loc, κ_skew, clock)
    κ_symmetricᶜᶜᶠ = κᶜᶜᶠ(i, j, k, grid, issd_coefficient_loc, κ_symmetric, clock)

    # Average... of... the gradient!
    ∂x_c = ℑxzᶜᵃᶠ(i, j, k, grid, ∂xᶠᶜᶜ, c)
    ∂y_c = ℑyzᵃᶜᶠ(i, j, k, grid, ∂yᶜᶠᶜ, c)

    R₃₁ = isopycnal_rotation_tensor_xz_ccf(i, j, k, grid, buoyancy, fields, closure.isopycnal_tensor)
    R₃₂ = isopycnal_rotation_tensor_yz_ccf(i, j, k, grid, buoyancy, fields, closure.isopycnal_tensor)

    κ_symmetric_∂z_c = explicit_κ_∂z_c(i, j, k, grid, TD(), c, κ_symmetricᶜᶜᶠ, closure, buoyancy, fields)

    ϵ = tapering_factorᶜᶜᶠ(i, j, k, grid, closure, fields, buoyancy)
    
    return - ϵ * κ_symmetric_∂z_c - ϵ * ((κ_symmetricᶜᶜᶠ + κ_skewᶜᶜᶠ) * R₃₁ * ∂x_c +
                                         (κ_symmetricᶜᶜᶠ + κ_skewᶜᶜᶠ) * R₃₂ * ∂y_c)
end
=#

@inline function explicit_κ_∂z_c(i, j, k, grid, ::ExplicitTimeDiscretization, κ_symmetricᶜᶜᶠ, closure, buoyancy, tracers)
    ∂z_c = ∂zᶜᶜᶠ(i, j, k, grid, c)
    R₃₃ = isopycnal_rotation_tensor_zz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    ϵ = tapering_factorᶜᶜᶠ(i, j, k, grid, closure, tracers, buoyancy)
    return ϵ * κ_symmetricᶜᶜᶠ * R₃₃ * ∂z_c
end

@inline explicit_R₃₃_∂z_c(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, args...) = zero(grid)
@inline explicit_R₃₃_∂z_u(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, args...) = zero(grid)
@inline explicit_R₃₃_∂z_v(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, args...) = zero(grid)

@inline function κzᶜᶜᶠ(i, j, k, grid, closure::FlavorOfTISSD, K, ::Val{id}, clock) where id
    closure = getclosure(i, j, closure)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, id)
    ϵ_R₃₃ = @inbounds K.ϵ_R₃₃[i, j, k] # tapered 33 component of rotation tensor
    return ϵ_R₃₃ * κᶜᶜᶠ(i, j, k, grid, issd_coefficient_loc, κ_symmetric, clock)
end

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
              
