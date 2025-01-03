using Oceananigans.Fields: VelocityFields

struct AdvectiveFormulation end
struct DiffusiveFormulation end

struct IsopycnalSkewSymmetricDiffusivity{TD, A, K, S, M, L, N} <: AbstractTurbulenceClosure{TD, N}
                    κ_skew :: K
               κ_symmetric :: S
          isopycnal_tensor :: M
             slope_limiter :: L
    
    function IsopycnalSkewSymmetricDiffusivity{TD, A, N}(κ_skew :: K,
                                                         κ_symmetric :: S,
                                                         isopycnal_tensor :: I,
                                                         slope_limiter :: L) where {TD, A, K, S, I, L, N}

        return new{TD, A, K, S, I, L, N}(κ_skew, κ_symmetric, isopycnal_tensor, slope_limiter)
    end
end

const ISSD{TD, A} = IsopycnalSkewSymmetricDiffusivity{TD, A} where {TD, A}
const ISSDVector{TD, A} = AbstractVector{<:ISSD{TD, A}} where {TD, A}
const FlavorOfISSD{TD, A} = Union{ISSD{TD, A}, ISSDVector{TD, A}} where {TD, A}
const SkewAdvectionISSD = ISSD{<:Any, <:AdvectiveFormulation}

# An ISSD type for which diffusive_flux_x, diffusive_flux_y, and diffusive_flux_z are all zero
const NoDiffusionISSD = ISSD{<:Any, <:AdvectiveFormulation, <:Any, Nothing}

# An ISSD type that does not have skew advection
const NoSkewAdvectionISSD = ISSD{<:Any, <:AdvectiveFormulation, Nothing}

const issd_coefficient_loc = (Center(), Center(), Face())

"""
    IsopycnalSkewSymmetricDiffusivity([time_disc=VerticallyImplicitTimeDiscretization(), FT=Float64;]
                                      κ_skew = 0,
                                      κ_symmetric = 0,
                                      skew_flux_formulation = AdvectiveFormulation(),
                                      isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                      slope_limiter = FluxTapering(1e-2))

Return parameters for an isopycnal skew-symmetric tracer diffusivity with skew diffusivity
`κ_skew` and symmetric diffusivity `κ_symmetric` that uses an `isopycnal_tensor` model for
for calculating the isopycnal slopes, and (optionally) applying a `slope_limiter` to the
calculated isopycnal slope values. The skew fluxes can be computed using either the `AdvectiveFormulation` 
or the `DiffusiveFormulation`.
    
Both `κ_skew` and `κ_symmetric` may be constants, arrays, fields, or functions of `(x, y, z, t)`.
"""
function IsopycnalSkewSymmetricDiffusivity(time_disc::TD = VerticallyImplicitTimeDiscretization(), FT = Float64;
                                           κ_skew = nothing,
                                           κ_symmetric = nothing,
                                           skew_flux_formulation::A = AdvectiveFormulation(),
                                           isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                           slope_limiter = FluxTapering(1e-2),
                                           required_halo_size::Int = 1) where {TD, A}

    # For the moment, allow only one skew coefficient for all tracers
    # TODO: maybe generalize it?
    if κ_skew isa NamedTuple && skew_flux_formulation isa AdvectiveFormulation
        error("Only one skew coefficient for all tracers is currently supported with the AdvectiveFormulation.")
    end

    isopycnal_tensor isa SmallSlopeIsopycnalTensor ||
        error("Only isopycnal_tensor=SmallSlopeIsopycnalTensor() is currently supported.")

    return IsopycnalSkewSymmetricDiffusivity{TD, A, required_halo_size}(convert_diffusivity(FT, κ_skew),
                                                                        convert_diffusivity(FT, κ_symmetric),
                                                                        isopycnal_tensor,
                                                                        slope_limiter)
end

IsopycnalSkewSymmetricDiffusivity(FT::DataType; kw...) = 
    IsopycnalSkewSymmetricDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

function with_tracers(tracers, closure::ISSD{TD, A, N}) where {TD, A<:DiffusiveFormulation, N}
    κ_skew = !isa(closure.κ_skew, NamedTuple) ? closure.κ_skew : tracer_diffusivities(tracers, closure.κ_skew)
    κ_symmetric = !isa(closure.κ_symmetric, NamedTuple) ? closure.κ_symmetric : tracer_diffusivities(tracers, closure.κ_symmetric)
    return IsopycnalSkewSymmetricDiffusivity{TD, A, N}(κ_skew, κ_symmetric, closure.isopycnal_tensor, closure.slope_limiter)
end

function with_tracers(tracers, closure::ISSD{TD, A, N}) where {TD, A<:AdvectiveFormulation, N}
    κ_skew = closure.κ_skew
    κ_symmetric = !isa(closure.κ_symmetric, NamedTuple) ? closure.κ_symmetric : tracer_diffusivities(tracers, closure.κ_symmetric)
    return IsopycnalSkewSymmetricDiffusivity{TD, A, N}(κ_skew, κ_symmetric, closure.isopycnal_tensor, closure.slope_limiter)
end

# For ensembles of closures
function with_tracers(tracers, closure_vector::ISSDVector)
    arch = architecture(closure_vector)

    if arch isa Architectures.GPU
        closure_vector = Vector(closure_vector)
    end

    Ex = length(closure_vector)
    closure_vector = [with_tracers(tracers, closure_vector[i]) for i=1:Ex]

    return on_architecture(arch, closure_vector)
end

function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfISSD{TD, A}) where {TD, A}
    if TD() isa VerticallyImplicitTimeDiscretization
        # Precompute the _tapered_ 33 component of the isopycnal rotation tensor
        diffusivities = (; ϵ_R₃₃ = Field((Center, Center, Face), grid))
    else
        diffusivities = NamedFieldTuple()
    end

    if A() isa AdvectiveFormulation && !(closure.κ_skew isa Nothing)
        U = VelocityFields(grid)
        diffusivities = merge(diffusivities, U)
    end

    return diffusivities
end

function compute_diffusivities!(diffusivities, closure::FlavorOfISSD, model; parameters = :xyz)

    arch = model.architecture
    grid = model.grid
    tracers = model.tracers
    buoyancy = model.buoyancy

    launch!(arch, grid, parameters,
            compute_tapered_R₃₃!, diffusivities.ϵ_R₃₃, grid, closure, tracers, buoyancy)


    compute_eddy_velocities!(diffusivities, closure, model; parameters)

    return nothing
end

@kernel function compute_tapered_R₃₃!(ϵ_R₃₃, grid, closure, tracers, buoyancy) 
    i, j, k, = @index(Global, NTuple)

    closure = getclosure(i, j, closure)
    R₃₃ = isopycnal_rotation_tensor_zz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    ϵ = tapering_factorᶜᶜᶠ(i, j, k, grid, closure, tracers, buoyancy)

    @inbounds ϵ_R₃₃[i, j, k] = ϵ * R₃₃
end

#####
##### Tapering
#####

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

# Make sure we do not need to perform heavy calculations if we really do not need to
@inline diffusive_flux_x(i, j, k, grid, ::NoDiffusionISSD, K, ::Val{tracer_index}, args...) where tracer_index = zero(grid)
@inline diffusive_flux_y(i, j, k, grid, ::NoDiffusionISSD, K, ::Val{tracer_index}, args...) where tracer_index = zero(grid)
@inline diffusive_flux_z(i, j, k, grid, ::NoDiffusionISSD, K, ::Val{tracer_index}, args...) where tracer_index = zero(grid)

# Diffusive fluxes
@inline get_tracer_κ(κ::NamedTuple, grid, tracer_index) = @inbounds κ[tracer_index]
@inline get_tracer_κ(::Nothing, grid, tracer_index) = zero(grid)
@inline get_tracer_κ(κ, grid, tracer_index) = κ

# Remove skew coefficient if we are using the advective formulation
@inline skew_diffusivity(i, j, k, grid, closure, κ, args...) = κ(i, j, k, grid, args...)
@inline skew_diffusivity(i, j, k, grid, ::SkewAdvectionISSD, args...) = zero(grid)

# defined at fcc
@inline function diffusive_flux_x(i, j, k, grid,
                                  closure::Union{ISSD, ISSDVector}, diffusivity_fields, ::Val{tracer_index},
                                  c, clock, fields, buoyancy) where tracer_index

    closure = getclosure(i, j, closure)

    κ_skew = get_tracer_κ(closure.κ_skew, grid, tracer_index)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, grid, tracer_index)

    κ_skewᶠᶜᶜ = skew_diffusivity(i, j, k, grid, closure, κᶠᶜᶜ, issd_coefficient_loc, κ_skew, clock)
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
                                  closure::Union{ISSD, ISSDVector}, diffusivity_fields, ::Val{tracer_index},
                                  c, clock, fields, buoyancy) where tracer_index

    closure = getclosure(i, j, closure)

    κ_skew = get_tracer_κ(closure.κ_skew, grid, tracer_index)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, grid, tracer_index)

    κ_skewᶜᶠᶜ = skew_diffusivity(i, j, k, grid, closure, κᶜᶠᶜ, issd_coefficient_loc, κ_skew, clock)
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
                                  closure::FlavorOfISSD{TD}, diffusivity_fields, ::Val{tracer_index},
                                  c, clock, fields, buoyancy) where {tracer_index, TD}

    closure = getclosure(i, j, closure)

    κ_skew = get_tracer_κ(closure.κ_skew, grid, tracer_index)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, grid, tracer_index)

    κ_skewᶜᶜᶠ = skew_diffusivity(i, j, k, grid, closure, κᶜᶜᶠ, issd_coefficient_loc, κ_skew, clock)
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

@inline function explicit_κ_∂z_c(i, j, k, grid, ::ExplicitTimeDiscretization, κ_symmetricᶜᶜᶠ, closure, buoyancy, tracers)
    ∂z_c = ∂zᶜᶜᶠ(i, j, k, grid, c)
    R₃₃ = isopycnal_rotation_tensor_zz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    
    ϵ = tapering_factorᶜᶜᶠ(i, j, k, grid, closure, tracers, buoyancy)

    return ϵ * κ_symmetricᶜᶜᶠ * R₃₃ * ∂z_c
end

@inline explicit_κ_∂z_c(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, args...) = zero(grid)

@inline function κzᶜᶜᶠ(i, j, k, grid, closure::FlavorOfISSD, K, ::Val{id}, clock) where id
    closure = getclosure(i, j, closure)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, grid, id)
    ϵ_R₃₃ = @inbounds K.ϵ_R₃₃[i, j, k] # tapered 33 component of rotation tensor
    return ϵ_R₃₃ * κᶜᶜᶠ(i, j, k, grid, issd_coefficient_loc, κ_symmetric, clock)
end

@inline viscous_flux_ux(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)
@inline viscous_flux_uy(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)
@inline viscous_flux_uz(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)

@inline viscous_flux_vx(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)
@inline viscous_flux_vy(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)
@inline viscous_flux_vz(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)

@inline viscous_flux_wx(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)
@inline viscous_flux_wy(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)
@inline viscous_flux_wz(i, j, k, grid, closure::Union{ISSD, ISSDVector}, args...) = zero(grid)

#####
##### Show
#####

Base.summary(closure::ISSD) = string("IsopycnalSkewSymmetricDiffusivity",
                                     "(κ_skew=",
                                     prettysummary(closure.κ_skew),
                                     ", κ_symmetric=", prettysummary(closure.κ_symmetric), ")")

Base.show(io::IO, closure::ISSD) =
    print(io, "IsopycnalSkewSymmetricDiffusivity: " *
              "(κ_symmetric=$(closure.κ_symmetric), κ_skew=$(closure.κ_skew), " *
              "(isopycnal_tensor=$(closure.isopycnal_tensor), slope_limiter=$(closure.slope_limiter))")
              

function compute_diffusivities!(diffusivities, closure::FlavorOfTISSD{TD}, model; parameters = :xyz) where TD

    arch = model.architecture
    grid = model.grid
    clock = model.clock
    tracers = model.tracers
    buoyancy = model.buoyancy

    if TD() isa VerticallyImplicitTimeDiscretization
        launch!(arch, grid, parameters,
                triad_compute_tapered_R₃₃!,
                diffusivities, grid, closure, clock, buoyancy, tracers)
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
    bx = ∂x_b(ix, j, kx, grid, buoyancy, tracers)
    bz = ∂z_b(iz, j, kz, grid, buoyancy, tracers)
    bz = max(bz, zero(grid))
    return ifelse(bz == 0, zero(grid), - bx / bz)
end

@inline function triad_Sy(i, jy, jz, ky, kz, grid, buoyancy, tracers)
    by = ∂y_b(i, jy, ky, grid, buoyancy, tracers)
    bz = ∂z_b(i, jz, kz, grid, buoyancy, tracers)
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

@inline ϵκx⁺⁺(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_x(i+1, i, j, k, k+1, grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)
@inline ϵκx⁺⁻(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_x(i+1, i, j, k, k,   grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)
@inline ϵκx⁻⁺(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_x(i,   i, j, k, k+1, grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)
@inline ϵκx⁻⁻(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_x(i,   i, j, k, k,   grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)

@inline ϵκy⁺⁺(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_y(i, j+1, j, k, k+1, grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)
@inline ϵκy⁺⁻(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_y(i, j+1, j, k, k,   grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)
@inline ϵκy⁻⁺(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_y(i, j,   j, k, k+1, grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)
@inline ϵκy⁻⁻(i, j, k, grid, loc, κ, clock, sl, b, C) = triad_mask_y(i, j,   j, k, k,   grid) * κᶜᶜᶜ(i, j, k, grid, loc, κ, clock) * tapering_factorᶜᶜᶜ(i, j, k, grid, sl, b, C)

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
    ∂x_c = ∂xᶠᶜᶜ(i, j, k, grid, c)

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

    ∂y_c = ∂yᶜᶠᶜ(i, j, k, grid, c)

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
    
    κR₃₁_∂x_c = (ϵκˣ⁻⁻ * Sx⁻⁻(i, j, k,   grid, b, C) * ∂xᶠᶜᶜ(i,   j, k,   grid, c) +
                 ϵκˣ⁺⁻ * Sx⁺⁻(i, j, k,   grid, b, C) * ∂xᶠᶜᶜ(i+1, j, k,   grid, c) +
                 ϵκˣ⁻⁺ * Sx⁻⁺(i, j, k-1, grid, b, C) * ∂xᶠᶜᶜ(i,   j, k-1, grid, c) +
                 ϵκˣ⁺⁺ * Sx⁺⁺(i, j, k-1, grid, b, C) * ∂xᶠᶜᶜ(i+1, j, k-1, grid, c)) / 4

    κR₃₂_∂y_c = (ϵκʸ⁻⁻ * Sy⁻⁻(i, j, k,   grid, b, C) * ∂yᶜᶠᶜ(i, j,   k,   grid, c) +
                 ϵκʸ⁺⁻ * Sy⁺⁻(i, j, k,   grid, b, C) * ∂yᶜᶠᶜ(i, j+1, k,   grid, c) +
                 ϵκʸ⁻⁺ * Sy⁻⁺(i, j, k-1, grid, b, C) * ∂yᶜᶠᶜ(i, j,   k-1, grid, c) +
                 ϵκʸ⁺⁺ * Sy⁺⁺(i, j, k-1, grid, b, C) * ∂yᶜᶠᶜ(i, j+1, k-1, grid, c)) / 4

    κϵ_R₃₃_∂z_c = explicit_R₃₃_∂z_c(i, j, k, grid, TD(), c, closure, b, C)

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

@inline function explicit_R₃₃_∂z_c(i, j, k, grid, ::ExplicitTimeDiscretization, c, closure, b, C) 
    κ  = closure.κ_symmetric
    sl = closure.slope_limiter
    return ϵκR₃₃(i, j, k, grid, κ, clock, sl, b, C) * ∂zᶜᶜᶠ(i, j, k, grid, c)
end

@inline explicit_R₃₃_∂z_c(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, c, closure, b, C) = zero(grid)

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

@inline function mask_inactive_points_ℑxzᶜᵃᶜ(i, j, k, grid, f::Function, args...) 
    neighboring_active_nodes = ℑxzᶜᵃᶜ(i, j, k, grid, not_peripheral_node, Face(), Center(), Face())
    return ifelse(neighboring_active_nodes == 0, zero(grid),
                  ℑxzᶜᵃᶜ(i, j, k, grid, f, args...) / neighboring_active_nodes)
end

@inline function mask_inactive_points_ℑyzᵃᶜᶜ(i, j, k, grid, f::Function, args...) 
    neighboring_active_nodes = @inbounds ℑyzᵃᶜᶜ(i, j, k, grid, not_peripheral_node, Center(), Face(), Face())
    return ifelse(neighboring_active_nodes == 0, zero(grid),
                  ℑyzᵃᶜᶜ(i, j, k, grid, f, args...) / neighboring_active_nodes)
end

# the `tapering_factor` function as well as the slope function `Sxᶠᶜᶠ` and `Syᶜᶠᶠ`
# are defined in the `advective_skew_diffusion.jl` file
@inline function tapering_factorᶜᶜᶜ(i, j, k, grid, slope_limiter, buoyancy, tracers)
    Sx = mask_inactive_points_ℑxzᶜᵃᶜ(i, j, k, grid, Sxᶠᶜᶠ, buoyancy, tracers)
    Sy = mask_inactive_points_ℑyzᵃᶜᶜ(i, j, k, grid, Syᶜᶠᶠ, buoyancy, tracers)
    return tapering_factor(Sx, Sy, slope_limiter)
end
