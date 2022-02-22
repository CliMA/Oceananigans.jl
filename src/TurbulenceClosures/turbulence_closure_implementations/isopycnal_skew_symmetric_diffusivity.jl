struct IsopycnalSkewSymmetricDiffusivity{K, S, M, L} <: AbstractTurbulenceClosure{Explicit}
                    κ_skew :: K
               κ_symmetric :: S
          isopycnal_tensor :: M
             slope_limiter :: L
    
    function IsopycnalSkewSymmetricDiffusivity(κ_skew::K, κ_symmetric::S, isopycnal_tensor::I, slope_limiter::L) where {K, S, I, L}

        isopycnal_tensor isa SmallSlopeIsopycnalTensor ||
            error("Only isopycnal_tensor=SmallSlopeIsopycnalTensor() is currently supported.")

        return new{K, S, I, L}(κ_skew, κ_symmetric, isopycnal_tensor, slope_limiter)
    end
end

const ISSD = IsopycnalSkewSymmetricDiffusivity

ISSDVector = AbstractVector{<:ISSD}

"""
    IsopycnalSkewSymmetricDiffusivity([FT=Float64;]
                                      κ_skew = 0,
                                      κ_symmetric = 0,
                                      isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                      slope_limiter = nothing)

Return parameters for an isopycnal skew-symmetric tracer diffusivity with skew diffusivity
`κ_skew` and symmetric diffusivity `κ_symmetric` that uses an `isopycnal_tensor` model for
for calculating the isopycnal slopes, and (optionally) applying a `slope_limiter` to the
calculated isopycnal slope values.
    
Both `κ_skew` and `κ_symmetric` may be constants, arrays, fields, or functions of `(x, y, z, t)`.
"""
IsopycnalSkewSymmetricDiffusivity(FT=Float64; κ_skew=0, κ_symmetric=0, isopycnal_tensor=SmallSlopeIsopycnalTensor(), slope_limiter=nothing) =
    IsopycnalSkewSymmetricDiffusivity(convert_diffusivity(FT, κ_skew, Val(false)), convert_diffusivity(FT, κ_symmetric, Val(false)), isopycnal_tensor, slope_limiter)

function with_tracers(tracers, closure::ISSD)
    κ_skew = !isa(closure.κ_skew, NamedTuple) ? closure.κ_skew : tracer_diffusivities(tracers, closure.κ_skew)
    κ_symmetric = !isa(closure.κ_symmetric, NamedTuple) ? closure.κ_symmetric : tracer_diffusivities(tracers, closure.κ_symmetric)
    return IsopycnalSkewSymmetricDiffusivity(κ_skew, κ_symmetric, closure.isopycnal_tensor, closure.slope_limiter)
end

# For ensembles of closures
function with_tracers(tracers, closure_vector::ISSDVector)
    arch = architecture(closure_vector)

    if arch isa Architectures.GPU
        closure_vector = Vector(closure_vector)
    end

    Ex = length(closure_vector)
    closure_vector = [with_tracers(tracers, closure_vector[i]) for i=1:Ex]

    return arch_array(arch, closure_vector)
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
@inline function taper_factor_ccc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, tapering::FluxTapering) where FT
    bx = ℑxᶜᵃᵃ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    by = ℑyᵃᶜᵃ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    
    slope_x = - bx / bz
    slope_y = - by / bz
    slope² = ifelse(bz <= 0, zero(FT), slope_x^2 + slope_y^2)

    return min(one(FT), tapering.max_slope^2 / slope²)
end

"""
    taper_factor_ccc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, ::Nothing) where FT

Returns 1 for the  isopycnal slope tapering factor, that is, no tapering is done.
"""
taper_factor_ccc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, ::Nothing) where FT = one(FT)

# Diffusive fluxes

@inline get_tracer_κ(κ::NamedTuple, tracer_index) = @inbounds κ[tracer_index]
@inline get_tracer_κ(κ, tracer_index) = κ

# defined at fcc
@inline function diffusive_flux_x(i, j, k, grid,
                                  closure::Union{ISSD, ISSDVector}, c, ::Val{tracer_index}, clock,
                                  diffusivity_fields, tracers, buoyancy, velocities) where tracer_index

    closure = get_closure_i(i, closure)

    κ_skew = get_tracer_κ(closure.κ_skew, tracer_index)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, tracer_index)

    κ_skewᶠᶜᶜ = κᶠᶜᶜ(i, j, k, grid, clock, κ_skew)
    κ_symmetricᶠᶜᶜ = κᶠᶜᶜ(i, j, k, grid, clock, κ_symmetric)

    ∂x_c = ∂xᶠᶜᶜ(i, j, k, grid, c)
    ∂y_c = ℑxyᶠᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, c)
    ∂z_c = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, c)

    R₁₁ = one(eltype(grid))
    R₁₂ = zero(eltype(grid))
    R₁₃ = isopycnal_rotation_tensor_xz_fcc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    
    ϵ = taper_factor_ccc(i, j, k, grid, buoyancy, tracers, closure.slope_limiter)

    return - ϵ * (           κ_symmetricᶠᶜᶜ * R₁₁ * ∂x_c +
                             κ_symmetricᶠᶜᶜ * R₁₂ * ∂y_c +
                  (κ_symmetricᶠᶜᶜ - κ_skewᶠᶜᶜ) * R₁₃ * ∂z_c)
end

# defined at cfc
@inline function diffusive_flux_y(i, j, k, grid,
                                  closure::Union{ISSD, ISSDVector}, c, ::Val{tracer_index}, clock,
                                  diffusivity_fields, tracers, buoyancy, velocities) where tracer_index

    closure = get_closure_i(i, closure)

    κ_skew = get_tracer_κ(closure.κ_skew, tracer_index)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, tracer_index)

    κ_skewᶜᶠᶜ = κᶜᶠᶜ(i, j, k, grid, clock, κ_skew)
    κ_symmetricᶜᶠᶜ = κᶜᶠᶜ(i, j, k, grid, clock, κ_symmetric)

    ∂x_c = ℑxyᶜᶠᵃ(i, j, k, grid, ∂xᶠᶜᶜ, c)
    ∂y_c = ∂yᶜᶠᶜ(i, j, k, grid, c)
    ∂z_c = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, c)

    R₂₁ = zero(eltype(grid))
    R₂₂ = one(eltype(grid))
    R₂₃ = isopycnal_rotation_tensor_yz_cfc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    ϵ = taper_factor_ccc(i, j, k, grid, buoyancy, tracers, closure.slope_limiter)

    return - ϵ * (           κ_symmetricᶜᶠᶜ * R₂₁ * ∂x_c +
                             κ_symmetricᶜᶠᶜ * R₂₂ * ∂y_c +
                  (κ_symmetricᶜᶠᶜ - κ_skewᶜᶠᶜ) * R₂₃ * ∂z_c)
end

# defined at ccf
@inline function diffusive_flux_z(i, j, k, grid,
                                  closure::Union{ISSD, ISSDVector}, c, ::Val{tracer_index}, clock,
                                  diffusivity_fields, tracers, buoyancy, velocities) where tracer_index

    closure = get_closure_i(i, closure)

    κ_skew = get_tracer_κ(closure.κ_skew, tracer_index)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, tracer_index)

    κ_skewᶜᶜᶠ = κᶜᶜᶠ(i, j, k, grid, clock, κ_skew)
    κ_symmetricᶜᶜᶠ = κᶜᶜᶠ(i, j, k, grid, clock, κ_symmetric)

    ∂x_c = ℑxzᶜᵃᶠ(i, j, k, grid, ∂xᶠᶜᶜ, c)
    ∂y_c = ℑyzᵃᶜᶠ(i, j, k, grid, ∂yᶜᶠᶜ, c)
    ∂z_c = ∂zᶜᶜᶠ(i, j, k, grid, c)

    R₃₁ = isopycnal_rotation_tensor_xz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    R₃₂ = isopycnal_rotation_tensor_yz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    R₃₃ = isopycnal_rotation_tensor_zz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    ϵ = taper_factor_ccc(i, j, k, grid, buoyancy, tracers, closure.slope_limiter)

    return - ϵ * ((κ_symmetricᶜᶜᶠ + κ_skewᶜᶜᶠ) * R₃₁ * ∂x_c +
                  (κ_symmetricᶜᶜᶠ + κ_skewᶜᶜᶠ) * R₃₂ * ∂y_c +
                                κ_symmetricᶜᶜᶠ * R₃₃ * ∂z_c)
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

#####
##### Show
#####

Base.show(io::IO, closure::ISSD) =
    print(io, "IsopycnalSkewSymmetricDiffusivity: " *
              "(κ_symmetric=$(closure.κ_symmetric), κ_skew=$(closure.κ_skew), " *
              "(isopycnal_tensor=$(closure.isopycnal_tensor), slope_limiter=$(closure.slope_limiter))")
              
