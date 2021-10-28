struct IsopycnalSkewSymmetricDiffusivity{K, S, M, L} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization}
             κ_skew :: K
        κ_symmetric :: S
    isopycnal_model :: M
      slope_limiter :: L
    
      IsopycnalSkewSymmetricDiffusivity(κ_skew, κ_symmetric, isopycnal_model, slope_limiter) = 
        typeof(isopycnal_model) == IsopycnalTensor ? 
        error("IsopycnalTensor not implemented yet; use SmallSlopeIsopycnalTensor instead.") : 
        new{typeof(κ_skew), typeof(κ_symmetric), typeof(isopycnal_model), typeof(slope_limiter)}(κ_skew, κ_symmetric, isopycnal_model, slope_limiter)
end

const ISSD = IsopycnalSkewSymmetricDiffusivity

ISSDVector = AbstractVector{<:ISSD}

"""
    IsopycnalSkewSymmetricDiffusivity([FT=Float64;] κ_skew=0, κ_symmetric=0,
                                      isopycnal_model=SmallSlopeIsopycnalTensor(), slope_limiter=nothing)

Return parameters for an isopycnal skew-symmetric tracer diffusivity with skew diffusivity
`κ_skew` and symmetric diffusivity `κ_symmetric` using an `isopycnal_model` for calculating
the isopycnal slopes, and optionally applying a `slope_limiter`. Both `κ_skew` and `κ_symmetric`
may be constants, arrays, fields, or functions of `(x, y, z, t)`.
"""
IsopycnalSkewSymmetricDiffusivity(FT=Float64; κ_skew=0, κ_symmetric=0, isopycnal_model=SmallSlopeIsopycnalTensor(), slope_limiter=nothing) =
    IsopycnalSkewSymmetricDiffusivity(convert_diffusivity(FT, κ_skew), convert_diffusivity(FT, κ_symmetric), isopycnal_model, slope_limiter)

function with_tracers(tracers, closure::ISSD)
    κ_skew = tracer_diffusivities(tracers, closure.κ_skew)
    κ_symmetric = tracer_diffusivities(tracers, closure.κ_symmetric)
    return IsopycnalSkewSymmetricDiffusivity(κ_skew, κ_symmetric, closure.isopycnal_model, closure.slope_limiter)
end

function with_tracers(tracers, closure_vector::ISSDVector)
    arch = architecture(closure_vector)
    Ex = length(closure_vector)
    return arch_array(arch, [with_tracers(tracers, closure_vector[i]) for i=1:Ex])
end

#####
##### Tapering
#####

struct FluxTapering{FT}
    max_slope :: FT
end

"""
    taper_factor_ccc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, tapering::FluxTapering)

Return the tapering factor `min(1, Sₘₐₓ² / slope²)`, where `slope² = slope_x² + slope_y²`
that multiplies all components of the isopycnal slope tensor. All slopes involved in the
tapering factor are computed at the cell centers.

References
==========
R. Gerdes, C. Koberle, and J. Willebrand. (1991), "The influence of numerical advection schemes
    on the results of ocean general circulation models", Clim. Dynamics, 5 (4), 211–226.
"""
taper_factor_ccc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, ::Nothing) where FT = one(FT)

@inline function taper_factor_ccc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, tapering::FluxTapering) where FT
    bx = ℑxᶜᵃᵃ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    by = ℑyᵃᶜᵃ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    
    slope_x = - bx / bz
    slope_y = - by / bz
    slope² = ifelse(bz <= 0, zero(FT), slope_x^2 + slope_y^2)

    return min(one(FT), tapering.max_slope^2 / slope²)
end

# Diffusive fluxes

# defined at fcc
@inline function diffusive_flux_x(i, j, k, grid,
                                  closure::Union{ISSD, ISSDVector}, c, ::Val{tracer_index}, clock,
                                  diffusivity_fields, tracers, buoyancy, velocities) where tracer_index

    closure = get_closure_i(i, closure)

    κ_skew = @inbounds κᶠᶜᶜ(i, j, k, grid, clock, closure.κ_skew[tracer_index])
    κ_symmetric = @inbounds κᶠᶜᶜ(i, j, k, grid, clock, closure.κ_symmetric[tracer_index])

    ∂x_c = ∂xᶠᵃᵃ(i, j, k, grid, c)
    ∂y_c = ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, c)
    ∂z_c = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, c)

    R₁₁ = one(eltype(grid))
    R₁₂ = zero(eltype(grid))
    R₁₃ = isopycnal_rotation_tensor_xz_fcc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_model)
    
    ϵ = taper_factor_ccc(i, j, k, grid, buoyancy, tracers, closure.slope_limiter)

    return - ϵ * (           κ_symmetric * R₁₁ * ∂x_c +
                             κ_symmetric * R₁₂ * ∂y_c +
                  (κ_symmetric - κ_skew) * R₁₃ * ∂z_c)
end

# defined at cfc
@inline function diffusive_flux_y(i, j, k, grid,
                                  closure::Union{ISSD, ISSDVector}, c, ::Val{tracer_index}, clock,
                                  diffusivity_fields, tracers, buoyancy, velocities) where tracer_index

    closure = get_closure_i(i, closure)

    κ_skew = @inbounds κᶜᶠᶜ(i, j, k, grid, clock, closure.κ_skew[tracer_index])
    κ_symmetric = @inbounds κᶜᶠᶜ(i, j, k, grid, clock, closure.κ_symmetric[tracer_index])

    ∂x_c = ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᵃᵃ, c)
    ∂y_c = ∂yᵃᶠᵃ(i, j, k, grid, c)
    ∂z_c = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᵃᵃᶠ, c)

    R₂₁ = zero(eltype(grid))
    R₂₂ = one(eltype(grid))
    R₂₃ = isopycnal_rotation_tensor_yz_cfc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_model)

    ϵ = taper_factor_ccc(i, j, k, grid, buoyancy, tracers, closure.slope_limiter)

    return - ϵ * (           κ_symmetric * R₂₁ * ∂x_c +
                             κ_symmetric * R₂₂ * ∂y_c +
                  (κ_symmetric - κ_skew) * R₂₃ * ∂z_c)
end

# defined at ccf
@inline function diffusive_flux_z(i, j, k, grid,
                                  closure::Union{ISSD, ISSDVector}, c, ::Val{tracer_index}, clock,
                                  diffusivity_fields, tracers, buoyancy, velocities) where tracer_index

    closure = get_closure_i(i, closure)

    κ_skew = @inbounds κᶜᶜᶠ(i, j, k, grid, clock, closure.κ_skew[tracer_index])
    κ_symmetric = @inbounds κᶜᶜᶠ(i, j, k, grid, clock, closure.κ_symmetric[tracer_index])

    ∂x_c = ℑxzᶜᵃᶠ(i, j, k, grid, ∂xᶠᵃᵃ, c)
    ∂y_c = ℑyzᵃᶜᶠ(i, j, k, grid, ∂yᵃᶠᵃ, c)
    ∂z_c = ∂zᵃᵃᶠ(i, j, k, grid, c)

    R₃₁ = isopycnal_rotation_tensor_xz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_model)
    R₃₂ = isopycnal_rotation_tensor_yz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_model)
    R₃₃ = isopycnal_rotation_tensor_zz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_model)

    ϵ = taper_factor_ccc(i, j, k, grid, buoyancy, tracers, closure.slope_limiter)

    return - ϵ * ((κ_symmetric + κ_skew) * R₃₁ * ∂x_c +
                  (κ_symmetric + κ_skew) * R₃₂ * ∂y_c +
                             κ_symmetric * R₃₃ * ∂z_c)
end

@inline viscous_flux_ux(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(eltype(grid))
@inline viscous_flux_uy(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(eltype(grid))
@inline viscous_flux_uz(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(eltype(grid))

@inline viscous_flux_vx(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(eltype(grid))
@inline viscous_flux_vy(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(eltype(grid))
@inline viscous_flux_vz(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(eltype(grid))

@inline viscous_flux_wx(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(eltype(grid))
@inline viscous_flux_wy(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(eltype(grid))
@inline viscous_flux_wz(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(eltype(grid))

calculate_diffusivities!(diffusivity_fields, closure::Union{ISSD, ISSDVector}, model) = nothing

DiffusivityFields(arch, grid, tracer_names, bcs, ::Union{ISSD, ISSDVector}) = nothing

#####
##### Show
#####

Base.show(io::IO, closure::ISSD) =
    print(io, "IsopycnalSkewSymmetricDiffusivity: " *
              "(κ_symmetric=$(closure.κ_symmetric), κ_skew=$(closure.κ_skew), " *
              "(isopycnal_model=$(closure.isopycnal_model), slope_limiter=$(closure.slope_limiter))")
              
