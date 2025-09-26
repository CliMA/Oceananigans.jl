struct IsopycnalDiffusivity{TD, K, L, N} <: AbstractTurbulenceClosure{TD, N}
    κ_symmetric :: K
    tapering :: L
    IsopycnalDiffusivity{TD, N}(κ_symmetric :: K, tapering :: L) where {TD, K, L, N} = 
        new{TD, K, L, N}(κ_symmetric, tapering)
end

const TISSD{TD} = IsopycnalDiffusivity{TD} where TD
const TISSDVector{TD} = AbstractVector{<:TISSD{TD}} where TD
const FlavorOfTISSD{TD} = Union{TISSD{TD}, TISSDVector{TD}} where TD

"""
    IsopycnalDiffusivity([time_disc=VerticallyImplicitTimeDiscretization(), FT=Float64;]
                                           κ_symmetric = 0,
                                           tapering = FluxTapering(1e-2),
                                           required_halo_size::Int = 1)

Return parameters for an isopycnal tracer diffusivity with a symmetric diffusivity `κ_symmetric` 
that (optionally) applies a `tapering` to the calculated isopycnal slope values.

`κ_symmetric` may be a constant, an array, a field, or a function of `(x, y, z, t)`.

The formulation follows Griffies et al. (1998)

References
==========
* Griffies, S. M., A. Gnanadesikan, R. C. Pacanowski, V. D. Larichev, J. K. Dukowicz, and R. D. Smith (1998) Isoneutral diffusion in a z-coordinate ocean model. _J. Phys. Oceanogr._, **28**, 805–830, doi:10.1175/1520-0485(1998)028<0805:IDIAZC>2.0.CO;2
"""
function IsopycnalDiffusivity(time_disc=VerticallyImplicitTimeDiscretization(), FT=Float64;
                              κ_symmetric = 0,
                              tapering = FluxTapering(1.0),
                              required_halo_size::Int = 1)

    TD = typeof(time_disc)

    return IsopycnalDiffusivity{TD, required_halo_size}(convert_diffusivity(FT, κ_symmetric),
                                                        tapering)
end

IsopycnalDiffusivity(FT::DataType; kw...) = 
    IsopycnalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

with_tracers(tracers, closure) = closure

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
function build_diffusivity_fields(grid, clock, tracer_names, bcs, ::FlavorOfTISSD{TD}) where TD
    if TD() isa VerticallyImplicitTimeDiscretization
        # Precompute the _tapered_ 33 component of the isopycnal rotation tensor
        K = (; ϵκR₃₃ = ZFaceField(grid))
    else
        return nothing
    end

    return K
end

compute_diffusivities!(diffusivities, closure::FlavorOfTISSD{<:VerticallyImplicitTimeDiscretization}, model; parameters = :xyz) = 
    launch!(architecture(model.grid), model.grid, parameters,
            triad_compute_tapered_R₃₃!,
            diffusivities, model.grid, closure, model.clock, model.buoyancy, model.tracers)

@kernel function triad_compute_tapered_R₃₃!(K, grid, closure, clock, b, C) 
    i, j, k, = @index(Global, NTuple)
    closure = getclosure(i, j, closure)
    κ  = closure.κ_symmetric
    sl = closure.tapering
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

# We remove triads that live on a boundary (immersed or top / bottom / north / south / east / west)
@inline triad_mask_x(ix, iz, j, kx, kz, grid) = 
   !peripheral_node(ix, j, kx, grid, Face(), Center(), Center()) & !peripheral_node(iz, j, kz, grid, Center(), Center(), Face()) 

@inline triad_mask_y(i, jy, jz, ky, kz, grid) = 
   !peripheral_node(i, jy, ky, grid, Center(), Face(), Center()) & !peripheral_node(i, jz, kz, grid, Center(), Center(), Face())

@inline function triad_Sx(ix, iz, j, kx, kz, grid, tapering, buoyancy, tracers)
    bx = ∂x_b(ix, j, kx, grid, buoyancy, tracers)
    bz = ∂z_b(iz, j, kz, grid, buoyancy, tracers)
    Sx = ifelse(bz == 0, zero(grid), - bx / bz) 
    ϵ  = tapering_factor(Sx, zero(grid), tapering)
    return ϵ * Sx * triad_mask_x(ix, iz, j, kx, kz, grid)
end

@inline function triad_Sy(i, jy, jz, ky, kz, grid, tapering, buoyancy, tracers)
    by = ∂y_b(i, jy, ky, grid, buoyancy, tracers)
    bz = ∂z_b(i, jz, kz, grid, buoyancy, tracers)
    Sy = ifelse(bz == 0, zero(grid), - by / bz)
    ϵ  = tapering_factor(zero(grid), Sy, tapering)
    return ϵ * Sy * triad_mask_y(i, jy, jz, ky, kz, grid)
end

@inline ϵSx⁺⁺(i, j, k, grid, tapering, buoyancy, tracers) = triad_Sx(i+1, i, j, k, k+1, grid, tapering, buoyancy, tracers)
@inline ϵSx⁺⁻(i, j, k, grid, tapering, buoyancy, tracers) = triad_Sx(i+1, i, j, k, k,   grid, tapering, buoyancy, tracers)
@inline ϵSx⁻⁺(i, j, k, grid, tapering, buoyancy, tracers) = triad_Sx(i,   i, j, k, k+1, grid, tapering, buoyancy, tracers)
@inline ϵSx⁻⁻(i, j, k, grid, tapering, buoyancy, tracers) = triad_Sx(i,   i, j, k, k,   grid, tapering, buoyancy, tracers)

@inline ϵSy⁺⁺(i, j, k, grid, tapering, buoyancy, tracers) = triad_Sy(i, j+1, j, k, k+1, grid, tapering, buoyancy, tracers)
@inline ϵSy⁺⁻(i, j, k, grid, tapering, buoyancy, tracers) = triad_Sy(i, j+1, j, k, k,   grid, tapering, buoyancy, tracers)
@inline ϵSy⁻⁺(i, j, k, grid, tapering, buoyancy, tracers) = triad_Sy(i, j,   j, k, k+1, grid, tapering, buoyancy, tracers)
@inline ϵSy⁻⁻(i, j, k, grid, tapering, buoyancy, tracers) = triad_Sy(i, j,   j, k, k,   grid, tapering, buoyancy, tracers)

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
    sl = closure.tapering
    loc = (Center(), Center(), Center())

    κ⁺⁺ = κᶜᶜᶜ(i-1, j, k, grid, loc, κ, clock)
    κ⁺⁻ = κᶜᶜᶜ(i-1, j, k, grid, loc, κ, clock)
    κ⁻⁺ = κᶜᶜᶜ(i,   j, k, grid, loc, κ, clock)
    κ⁻⁻ = κᶜᶜᶜ(i,   j, k, grid, loc, κ, clock)

    # Small slope approximation
    ∂x_c = ∂xᶠᶜᶜ(i, j, k, grid, c)

    #       i-1     i 
    # k+1  -------------
    #           |      |
    #       ┏┗  ∘  ┛┓  | k
    #           |      |
    # k   ------|------|    

    Fx = (κ⁺⁺ * (∂x_c + ϵSx⁺⁺(i-1, j, k, grid, sl, b, C) * ∂zᶜᶜᶠ(i-1, j, k+1, grid, c)) +
          κ⁺⁻ * (∂x_c + ϵSx⁺⁻(i-1, j, k, grid, sl, b, C) * ∂zᶜᶜᶠ(i-1, j, k,   grid, c)) +
          κ⁻⁺ * (∂x_c + ϵSx⁻⁺(i,   j, k, grid, sl, b, C) * ∂zᶜᶜᶠ(i,   j, k+1, grid, c)) +
          κ⁻⁻ * (∂x_c + ϵSx⁻⁻(i,   j, k, grid, sl, b, C) * ∂zᶜᶜᶠ(i,   j, k,   grid, c))) / 4
    
    return - Fx
end

# defined at cfc
@inline function diffusive_flux_y(i, j, k, grid, closure::FlavorOfTISSD, K, ::Val{id},
                                  c, clock, C, b) where id

    closure = getclosure(i, j, closure)
    κ  = closure.κ_symmetric
    sl = closure.tapering
    loc = (Center(), Center(), Center())

    ∂y_c = ∂yᶜᶠᶜ(i, j, k, grid, c)

    κ⁺⁺ = κᶜᶜᶜ(i, j-1, k, grid, loc, κ, clock)
    κ⁺⁻ = κᶜᶜᶜ(i, j-1, k, grid, loc, κ, clock)
    κ⁻⁺ = κᶜᶜᶜ(i, j,   k, grid, loc, κ, clock)
    κ⁻⁻ = κᶜᶜᶜ(i, j,   k, grid, loc, κ, clock)
    
    Fy = (κ⁺⁺ * (∂y_c + ϵSy⁺⁺(i, j-1, k, grid, sl, b, C) * ∂zᶜᶜᶠ(i, j-1, k+1, grid, c)) +
          κ⁺⁻ * (∂y_c + ϵSy⁺⁻(i, j-1, k, grid, sl, b, C) * ∂zᶜᶜᶠ(i, j-1, k,   grid, c)) +
          κ⁻⁺ * (∂y_c + ϵSy⁻⁺(i, j,   k, grid, sl, b, C) * ∂zᶜᶜᶠ(i, j,   k+1, grid, c)) +
          κ⁻⁻ * (∂y_c + ϵSy⁻⁻(i, j,   k, grid, sl, b, C) * ∂zᶜᶜᶠ(i, j,   k,   grid, c))) / 4

    return - Fy
end

# defined at ccf
@inline function diffusive_flux_z(i, j, k, grid, closure::FlavorOfTISSD{TD}, K, ::Val{id},
                                  c, clock, C, b) where {TD, id}

    closure = getclosure(i, j, closure)
    κ  = closure.κ_symmetric
    sl = closure.tapering

    loc = (Center(), Center(), Center())

    κ⁻⁻ = κᶜᶜᶜ(i, j, k,   grid, loc, κ, clock)
    κ⁺⁻ = κᶜᶜᶜ(i, j, k,   grid, loc, κ, clock)
    κ⁻⁺ = κᶜᶜᶜ(i, j, k-1, grid, loc, κ, clock)
    κ⁺⁺ = κᶜᶜᶜ(i, j, k-1, grid, loc, κ, clock)

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
    
    κR₃₁_∂x_c = (κ⁻⁻ * ϵSx⁻⁻(i, j, k,   grid, sl, b, C) * ∂xᶠᶜᶜ(i,   j, k,   grid, c) +
                 κ⁺⁻ * ϵSx⁺⁻(i, j, k,   grid, sl, b, C) * ∂xᶠᶜᶜ(i+1, j, k,   grid, c) +
                 κ⁻⁺ * ϵSx⁻⁺(i, j, k-1, grid, sl, b, C) * ∂xᶠᶜᶜ(i,   j, k-1, grid, c) +
                 κ⁺⁺ * ϵSx⁺⁺(i, j, k-1, grid, sl, b, C) * ∂xᶠᶜᶜ(i+1, j, k-1, grid, c)) / 4

    κR₃₂_∂y_c = (κ⁻⁻ * ϵSy⁻⁻(i, j, k,   grid, sl, b, C) * ∂yᶜᶠᶜ(i, j,   k,   grid, c) +
                 κ⁺⁻ * ϵSy⁺⁻(i, j, k,   grid, sl, b, C) * ∂yᶜᶠᶜ(i, j+1, k,   grid, c) +
                 κ⁻⁺ * ϵSy⁻⁺(i, j, k-1, grid, sl, b, C) * ∂yᶜᶠᶜ(i, j,   k-1, grid, c) +
                 κ⁺⁺ * ϵSy⁺⁺(i, j, k-1, grid, sl, b, C) * ∂yᶜᶠᶜ(i, j+1, k-1, grid, c)) / 4

    κϵ_R₃₃_∂z_c = explicit_R₃₃_∂z_c(i, j, k, grid, TD(), c, clock, closure, b, C)

    return - κR₃₁_∂x_c - κR₃₂_∂y_c - κϵ_R₃₃_∂z_c
end

@inline function ϵκR₃₃(i, j, k, grid, κ, clock, sl, b, C) 
    loc = (Center(), Center(), Center())

    κ⁻⁻ = κᶜᶜᶜ(i, j, k,   grid, loc, κ, clock)
    κ⁺⁻ = κᶜᶜᶜ(i, j, k,   grid, loc, κ, clock)
    κ⁻⁺ = κᶜᶜᶜ(i, j, k-1, grid, loc, κ, clock)
    κ⁺⁺ = κᶜᶜᶜ(i, j, k-1, grid, loc, κ, clock)

    ϵκR₃₃ = (κ⁻⁻ * ϵSx⁻⁻(i, j, k,   grid, sl, b, C)^2 + κ⁻⁻ * ϵSy⁻⁻(i, j, k,   grid, sl, b, C)^2 +
             κ⁺⁻ * ϵSx⁺⁻(i, j, k,   grid, sl, b, C)^2 + κ⁺⁻ * ϵSy⁺⁻(i, j, k,   grid, sl, b, C)^2 +
             κ⁻⁺ * ϵSx⁻⁺(i, j, k-1, grid, sl, b, C)^2 + κ⁻⁺ * ϵSy⁻⁺(i, j, k-1, grid, sl, b, C)^2 +
             κ⁺⁺ * ϵSx⁺⁺(i, j, k-1, grid, sl, b, C)^2 + κ⁺⁺ * ϵSy⁺⁺(i, j, k-1, grid, sl, b, C)^2) / 4 

    return ϵκR₃₃
end

@inline function explicit_R₃₃_∂z_c(i, j, k, grid, ::ExplicitTimeDiscretization, c, clock, closure, b, C) 
    κ  = closure.κ_symmetric
    sl = closure.tapering
    return ϵκR₃₃(i, j, k, grid, κ, clock, sl, b, C) * ∂zᶜᶜᶠ(i, j, k, grid, c)
end

@inline explicit_R₃₃_∂z_c(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, c, clock, closure, b, C) = zero(grid)

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

# Base.summary(closure::TISSD) = string("IsopycnalDiffusivity",
#                                      "(κ_skew=",
#                                      prettysummary(closure.κ_skew),
#                                      ", κ_symmetric=", prettysummary(closure.κ_symmetric), ")")

# Base.show(io::IO, closure::TISSD) =
#     print(io, "IsopycnalDiffusivity: " *
#               "(κ_symmetric=$(closure.κ_symmetric), κ_skew=$(closure.κ_skew), " *
#               "(isopycnal_tensor=$(closure.isopycnal_tensor), slope_limiter=$(closure.slope_limiter))")