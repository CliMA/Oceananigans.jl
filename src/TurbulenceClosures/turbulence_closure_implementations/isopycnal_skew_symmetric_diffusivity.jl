struct IsopycnalSkewSymmetricDiffusivity{K, S, M, L} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization}
             κ_skew :: K
        κ_symmetric :: S
    isopycnal_model :: M
      slope_limiter :: L
end

const ISSD = IsopycnalSkewSymmetricDiffusivity

"""
    IsopycnalSkewSymmetricDiffusivity([FT=Float64;] κ_skew=0, κ_symmetric=0, isopycnal_model=SmallSlopeIsopycnalTensor())

Returns parameters for an isopycnal skew-symmetric tracer diffusivity with skew diffusivity
`κ_skew` and symmetric diffusivity `κ_symmetric` using an `isopycnal_model` for calculating
the isopycnal slopes. Both `κ_skew` and `κ_symmetric` may be constants, arrays, fields, or
functions of `(x, y, z, t)`.
"""
IsopycnalSkewSymmetricDiffusivity(FT=Float64; κ_skew=0, κ_symmetric=0, isopycnal_model=SmallSlopeIsopycnalTensor()) =
    IsopycnalSkewSymmetricDiffusivity(convert_diffusivity(FT, κ_skew), convert_diffusivity(FT, κ_symmetric), isopycnal_model)

function with_tracers(tracers, closure::ISSD)
    κ_skew = tracer_diffusivities(tracers, closure.κ_skew)
    κ_symmetric = tracer_diffusivities(tracers, closure.κ_symmetric)
    return IsopycnalSkewSymmetricDiffusivity(κ_skew, κ_symmetric, closure.isopycnal_model)
end

#####
##### Tapering
#####

struct FluxTapering end

taper_factor_ccc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, ::Nothing) where FT = one(FT)

@inline function taper_factor_ccc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, tapering::FluxTapering) where FT
    bx = ℑxᶜᵃᵃ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    by = ℑyᵃᶜᵃ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ℑyᵃᶜᵃ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    
    slope_x = - bx / bz
    slope_y = - by / bz
    slope_h² = slope_x^2 + slope_y^2

    return min(one(FT), tapering.max_slope^2 / slope_h²)
end

# Diffusive fluxes

@inline function diffusive_flux_x(i, j, k, grid,
                                  closure::ISSD, c, ::Val{tracer_index}, clock,
                                  diffusivity_fields, tracers, buoyancy, velocities) where tracer_index

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

@inline function diffusive_flux_y(i, j, k, grid,
                                  closure::ISSD, c, ::Val{tracer_index}, clock,
                                  diffusivity_fields, tracers, buoyancy, velocities) where tracer_index

    κ_skew = @inbounds κᶜᶠᶜ(i, j, k, grid, clock, closure.κ_skew[tracer_index])
    κ_symmetric = @inbounds κᶜᶠᶜ(i, j, k, grid, clock, closure.κ_symmetric[tracer_index])

    ∂x_c = ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᵃᵃ, c)
    ∂y_c = ∂yᵃᶠᵃ(i, j, k, grid, c)
    ∂z_c = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᵃᵃᶠ, c)

    R₂₁ = zeros(eltype(grid))
    R₂₂ = one(eltype(grid))
    R₂₃ = isopycnal_rotation_tensor_yz_cfc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_model)

    ϵ = taper_factor_ccc(i, j, k, grid, buoyancy, tracers, closure.slope_limiter)

    return - ϵ * (           κ_symmetric * R₂₁ * ∂x_c +
                             κ_symmetric * R₂₂ * ∂y_c +
                  (κ_symmetric - κ_skew) * R₂₃ * ∂z_c)
end

@inline function diffusive_flux_z(i, j, k, grid,
                                  closure::ISSD, c, ::Val{tracer_index}, clock,
                                  diffusivity_fields, tracers, buoyancy, velocities) where tracer_index

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

@inline viscous_flux_ux(i, j, k, grid, closure::ISSD, args...) = zero(eltype(grid))
@inline viscous_flux_uy(i, j, k, grid, closure::ISSD, args...) = zero(eltype(grid))
@inline viscous_flux_uz(i, j, k, grid, closure::ISSD, args...) = zero(eltype(grid))

@inline viscous_flux_vx(i, j, k, grid, closure::ISSD, args...) = zero(eltype(grid))
@inline viscous_flux_vy(i, j, k, grid, closure::ISSD, args...) = zero(eltype(grid))
@inline viscous_flux_vz(i, j, k, grid, closure::ISSD, args...) = zero(eltype(grid))

@inline viscous_flux_wx(i, j, k, grid, closure::ISSD, args...) = zero(eltype(grid))
@inline viscous_flux_wy(i, j, k, grid, closure::ISSD, args...) = zero(eltype(grid))
@inline viscous_flux_wz(i, j, k, grid, closure::ISSD, args...) = zero(eltype(grid))

calculate_diffusivities!(diffusivity_fields, closure::ISSD, model) = nothing

DiffusivityFields(arch, grid, tracer_names, bcs, ::ISSD) = nothing
