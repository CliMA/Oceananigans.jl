struct IsopycnalSkewSymmetricDiffusivity{K, S} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization}
         κ_skew :: K
    κ_symmetric :: S
end

const ISSD = IsopycnalSkewSymmetricDiffusivity

IsopycnalSkewSymmetricDiffusivity(FT=Float64; κ_skew=0, κ_symmetric=0) = IsopycnalSkewSymmetricDiffusivity(κ_skew, κ_symmetric)

function with_tracers(tracers, closure::ISSD)
    κ_skew = tracer_diffusivities(tracers, closure.κ_skew)
    κ_symmetric = tracer_diffusivities(tracers, closure.κ_symmetric)
    return IsopycnalSkewSymmetricDiffusivity(κ_skew, κ_symmetric)
end

# Diffusive fluxes for Leith diffusivities

@inline function diffusive_flux_x(i, j, k, grid,
                                  closure::ISSD, c, ::Val{tracer_index}, clock,
                                  diffusivity_fields, tracers, buoyancy, velocities) where tracer_index

    κ_skew = @inbounds κᶠᶜᶜ(i, j, k, grid, clock, closure.κ_skew[tracer_index])
    κ_symmetric = @inbounds κᶠᶜᶜ(i, j, k, grid, clock, closure.κ_symmetric[tracer_index])

    ∂x_c = ∂xᶠᵃᵃ(i, j, k, grid, c)
    ∂z_c = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, c)

    R₁₃ = isopycnal_rotation_tensor_xz_fcc(i, j, k, grid, buoyancy, tracers)

    return - κ_symmetric * ∂x_c + (κ_skew - κ_symmetric) * R₁₃ * ∂z_c
end

@inline function diffusive_flux_y(i, j, k, grid,
                                  closure::ISSD, c, ::Val{tracer_index}, clock,
                                  diffusivity_fields, tracers, buoyancy, velocities) where tracer_index

    κ_skew = @inbounds κᶜᶠᶜ(i, j, k, grid, clock, closure.κ_skew[tracer_index])
    κ_symmetric = @inbounds κᶜᶠᶜ(i, j, k, grid, clock, closure.κ_symmetric[tracer_index])

    ∂y_c = ∂yᵃᶠᵃ(i, j, k, grid, c)
    ∂z_c = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᵃᵃᶠ, c)

    R₂₃ = isopycnal_rotation_tensor_yz_cfc(i, j, k, grid, buoyancy, tracers)

    return - κ_symmetric * ∂y_c + (κ_skew - κ_symmetric) * R₂₃ * ∂z_c
end

@inline function diffusive_flux_z(i, j, k, grid,
                                  closure::ISSD, c, ::Val{tracer_index}, clock,
                                  diffusivity_fields, tracers, buoyancy, velocities) where tracer_index

    κ_skew = @inbounds κᶜᶜᶠ(i, j, k, grid, clock, closure.κ_skew[tracer_index])
    κ_symmetric = @inbounds κᶜᶜᶠ(i, j, k, grid, clock, closure.κ_symmetric[tracer_index])

    ∂x_c = ℑxzᶜᵃᶠ(i, j, k, grid, ∂xᶠᵃᵃ, c)
    ∂y_c = ℑyzᵃᶜᶠ(i, j, k, grid, ∂yᵃᶠᵃ, c)
    ∂z_c = ∂zᵃᵃᶠ(i, j, k, grid, c)

    R₃₁ = isopycnal_rotation_tensor_xz_ccf(i, j, k, grid, buoyancy, tracers)
    R₃₂ = isopycnal_rotation_tensor_yz_ccf(i, j, k, grid, buoyancy, tracers)
    R₃₃ = isopycnal_rotation_tensor_zz_ccf(i, j, k, grid, buoyancy, tracers)

    return - (κ_symmetric + κ_skew) * R₃₁ * ∂x_c - (κ_symmetric + κ_skew) * R₃₂ * ∂y_c - κ_symmetric * R₃₃ * ∂z_c
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
