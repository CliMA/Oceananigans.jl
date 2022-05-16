using Oceananigans.Operators: ℑxzᶜᶜᶠ, ℑyzᶜᶜᶠ, ℑxyᶜᶠᶜ, ℑyzᶜᶠᶜ, ℑxyᶠᶜᶜ, ℑxzᶠᶜᶜ
using Oceananigans.Advection: AbstractAdvectionScheme, div_Uc
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions

struct DiffusiveSkewFluxScheme end

struct IsopycnalSkewSymmetricDiffusivity{TD, A, K, S, M, L} <: AbstractTurbulenceClosure{TD}
    κ_skew :: K
    κ_symmetric :: S
    isopycnal_tensor :: M
    slope_limiter :: L
    skew_flux_scheme :: A
    
    function IsopycnalSkewSymmetricDiffusivity{TD}(κ_skew :: K,
                                                   κ_symmetric :: S,
                                                   isopycnal_tensor :: I,
                                                   slope_limiter :: L, 
                                                   skew_flux_scheme :: A) where {TD, K, S, I, L, A}

        return new{TD, A, K, S, I, L}(κ_skew, κ_symmetric, isopycnal_tensor, slope_limiter, skew_flux_scheme)
    end
end

const ISSD{TD, A} = IsopycnalSkewSymmetricDiffusivity{TD, A} where {TD, A}
const ISSDVector{TD, A} = AbstractVector{<:ISSD{TD, A}} where {TD, A}
const FlavorOfISSD{TD, A} = Union{ISSD{TD, A}, ISSDVector{TD, A}} where {TD, A}
const AdvectiveISSD = FlavorOfISSD{TD, <:AbstractAdvectionScheme} where TD
const issd_coefficient_loc = (Center(), Center(), Center())

"""
    IsopycnalSkewSymmetricDiffusivity([time_disc=VerticallyImplicitTimeDiscretization(), FT=Float64;]
                                      κ_skew = 0,
                                      κ_symmetric = 0,
                                      isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                      slope_limiter = nothing,
                                      skew_flux_scheme = DiffusiveSkewFluxScheme())

Return parameters for an isopycnal skew-symmetric tracer diffusivity with skew diffusivity
`κ_skew` and symmetric diffusivity `κ_symmetric` that uses an `isopycnal_tensor` model for
for calculating the isopycnal slopes, and (optionally) applying a `slope_limiter` to the
calculated isopycnal slope values.
    
Both `κ_skew` and `κ_symmetric` may be constants, arrays, fields, or functions of `(x, y, z, t)`.
"""
function IsopycnalSkewSymmetricDiffusivity(time_disc::TD = VerticallyImplicitTimeDiscretization(), FT = Float64;
                                           κ_skew = 0,
                                           κ_symmetric = 0,
                                           isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                           slope_limiter = nothing,
                                           skew_flux_scheme = DiffusiveSkewFluxScheme()) where TD

    isopycnal_tensor isa SmallSlopeIsopycnalTensor ||
        error("Only isopycnal_tensor=SmallSlopeIsopycnalTensor() is currently supported.")

    !(skew_flux_scheme isa DiffusiveSkewFluxScheme) && κ_symmetric != 0 &&
        error("Cannot use an advective skew_flux_scheme with non-zero κ_symmetric.")

    return IsopycnalSkewSymmetricDiffusivity{TD}(convert_diffusivity(FT, κ_skew),
                                                 convert_diffusivity(FT, κ_symmetric),
                                                 isopycnal_tensor,
                                                 slope_limiter,
                                                 skew_flux_scheme)
end

IsopycnalSkewSymmetricDiffusivity(FT::DataType; kw...) = 
    IsopycnalSkewSymmetricDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

function with_tracers(tracers, closure::ISSD{TD}) where TD
    # Only accept single skew diffusivities right now
    # κ_skew = !isa(closure.κ_skew, NamedTuple) ? closure.κ_skew : tracer_diffusivities(tracers, closure.κ_skew)
    κ_symmetric = !isa(closure.κ_symmetric, NamedTuple) ? closure.κ_symmetric : tracer_diffusivities(tracers, closure.κ_symmetric)
    return IsopycnalSkewSymmetricDiffusivity{TD}(closure.κ_skew,
                                                 κ_symmetric,
                                                 closure.isopycnal_tensor,
                                                 closure.slope_limiter,
                                                 closure.skew_flux_scheme)
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
##### Advective fluxes...
#####

@inline function ψʸ_GM_cff(i, j, k, grid, closure, buoyancy, tracers)
    closure = getclosure(i, j, closure)
    Sʸ = isopycnal_rotation_tensor_yz_cff(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    κ_skewᶜᶠᶠ = νᶜᶠᶠ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_skew)
    return κ_skewᶜᶠᶠ * Sʸ
end

@inline function ψˣ_GM_fcf(i, j, k, grid, closure, buoyancy, tracers)
    closure = getclosure(i, j, closure)
    Sˣ = isopycnal_rotation_tensor_xz_fcf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    κ_skewᶠᶜᶠ = νᶠᶜᶠ(i, j, k, grid, clock, issd_coefficient_loc, closure.κ_skew)
    return κ_skewᶠᶜᶠ * Sˣ
end

@inline u★_GM(i, j, k, grid, closure, buoyancy, tracers) = - ∂zᶠᶜᶜ(i, j, k, grid, ψˣ_GM_fcf, closure, buoyancy, tracers)
@inline v★_GM(i, j, k, grid, closure, buoyancy, tracers) = - ∂zᶜᶠᶜ(i, j, k, grid, ψʸ_GM_cff, closure, buoyancy, tracers)
@inline w★_GM(i, j, k, grid, closure, buoyancy, tracers) =
    ∂xᶜᶜᶠ(i, j, k, grid, ψˣ_GM_fcf, closure, buoyancy, tracers) + 
    ∂yᶜᶜᶠ(i, j, k, grid, ψʸ_GM_cff, closure, buoyancy, tracers)

# Note: computing diffusivities at cell centers for now.
function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfISSD{TD}) where TD

    ϵ_R₃₃ = nothing
    u★ = nothing
    v★ = nothing
    w★ = nothing

    if TD() isa VerticallyImplicitTimeDiscretization
        # Precompute the _tapered_ 33 component of the isopycnal rotation tensor
        ϵ_R₃₃ = Field{Center, Center, Face}(grid)
    end

    if closure isa AdvectiveISSD
        u_bcs = regularize_field_boundary_conditions(FieldBoundaryConditions(), grid, :u)
        v_bcs = regularize_field_boundary_conditions(FieldBoundaryConditions(), grid, :v)
        w_bcs = regularize_field_boundary_conditions(FieldBoundaryConditions(), grid, :w)

        U★ = (u = Field{Face, Center, Center}(grid, boundary_conditions=u_bcs),
              v = Field{Center, Face, Center}(grid, boundary_conditions=v_bcs),
              w = Field{Center, Center, Face}(grid, boundary_conditions=w_bcs))
    end

    return (; ϵ_R₃₃, U★)
end

function calculate_diffusivities!(diffusivities, closure::FlavorOfISSD{TD}, model) where TD

    arch = model.architecture
    grid = model.grid
    tracers = model.tracers
    buoyancy = model.buoyancy

    if TD() isa VerticallyImplicitTimeDiscretization
        R³³_event = launch!(arch, grid, :xyz,
                            compute_tapered_R₃₃!, diffusivities.ϵ_R₃₃, grid, closure, tracers, buoyancy,
                            dependencies = device_event(arch))
    else
        R³³_event = NoneEvent()
    end

    if closure isa AdvectiveISSD
        u★_event = launch!(arch, grid, :xyz,
                           compute_U★!, diffusivities.U★, grid, closure, tracers, buoyancy,
                           dependencies = device_event(arch))
    else
        u★_event = NoneEvent()
    end

    wait(device(arch), MultiEvent((R³³_event, u★_event)))

    fill_halo_regions!(diffusivities.U★)

    return nothing
end

@kernel function compute_tapered_R₃₃!(ϵ_R₃₃, grid, closure, tracers, buoyancy)
    i, j, k, = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)
    ϵ = taper_factor_ccc(i, j, k, grid, buoyancy, tracers, closure_ij.slope_limiter)
    R₃₃ = isopycnal_rotation_tensor_zz_ccf(i, j, k, grid, buoyancy, tracers, closure_ij.isopycnal_tensor)

    @inbounds ϵ_R₃₃[i, j, k] = ϵ * R₃₃
end

@kernel function compute_U★!(U★, grid, closure, tracers, buoyancy)
    i, j, k, = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)

    @inbounds begin
        U★.u[i, j, k] = u★_GM(i, j, k, grid, closure_ij, buoyancy, tracers)
        U★.v[i, j, k] = v★_GM(i, j, k, grid, closure_ij, buoyancy, tracers)
        U★.w[i, j, k] = w★_GM(i, j, k, grid, closure_ij, buoyancy, tracers)
    end
end

@inline ∇_dot_qᶜ(i, j, k, grid, closure::AdvectiveISSD, K, ::Val{id}, velocities, tracers, args...) where id =
    div_Uc(i, j, k, grid, closure.skew_flux_scheme, K.U★, tracers[id])

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
@inline function taper_factor_ccc(i, j, k, grid, buoyancy, tracers, tapering::FluxTapering)
    # TODO: handle boundaries!
    bx = ℑxᶜᶜᶜ(i, j, k, grid, ∂x_b, buoyancy, tracers)
    by = ℑyᶜᶜᶜ(i, j, k, grid, ∂y_b, buoyancy, tracers)
    bz = ℑzᶜᶜᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)

    slope_x = - bx / bz
    slope_y = - by / bz
    slope² = ifelse(bz <= 0, zero(grid), slope_x^2 + slope_y^2)

    return min(one(grid), tapering.max_slope^2 / slope²)
end

"""
    taper_factor_ccc(i, j, k, grid::AbstractGrid{FT}, buoyancy, tracers, ::Nothing) where FT

Returns 1 for the  isopycnal slope tapering factor, that is, no tapering is done.
"""
taper_factor_ccc(i, j, k, grid, buoyancy, tracers, ::Nothing) = one(grid)

# Diffusive fluxes

@inline get_tracer_κ(κ::NamedTuple, tracer_index) = @inbounds κ[tracer_index]
@inline get_tracer_κ(κ, tracer_index) = κ

# defined at fcc
@inline function diffusive_flux_x(i, j, k, grid,
                                  closure::Union{ISSD, ISSDVector}, diffusivity_fields, ::Val{tracer_index},
                                  velocities, tracers, clock, buoyancy) where tracer_index

    c = tracers[tracer_index]
    closure = getclosure(i, j, closure)

    κ_skew = get_tracer_κ(closure.κ_skew, tracer_index)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, tracer_index)

    κ_skewᶠᶜᶜ = κᶠᶜᶜ(i, j, k, grid, clock, issd_coefficient_loc, κ_skew)
    κ_symmetricᶠᶜᶜ = κᶠᶜᶜ(i, j, k, grid, clock, issd_coefficient_loc, κ_symmetric)

    ∂x_c = ∂xᶠᶜᶜ(i, j, k, grid, c)

    # Average... of... the gradient!
    ∂y_c = ℑxyᶠᶜᶜ(i, j, k, grid, ∂yᶜᶠᶜ, c)
    ∂z_c = ℑxzᶠᶜᶜ(i, j, k, grid, ∂zᶜᶜᶠ, c)

    # Gradient of the average.
    #∂y_c = ∂yᶠᶜᶜ(i, j, k, grid, ℑxyᶠᶠᶜ, c)
    #∂z_c = ∂zᶠᶜᶜ(i, j, k, grid, ℑxzᶠᶜᶠ, c)

    R₁₁ = one(grid)
    R₁₂ = zero(grid)
    R₁₃ = isopycnal_rotation_tensor_xz_fcc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    
    ϵ = taper_factor_ccc(i, j, k, grid, buoyancy, tracers, closure.slope_limiter)

    return - ϵ * (              κ_symmetricᶠᶜᶜ * R₁₁ * ∂x_c +
                                κ_symmetricᶠᶜᶜ * R₁₂ * ∂y_c +
                  (κ_symmetricᶠᶜᶜ - κ_skewᶠᶜᶜ) * R₁₃ * ∂z_c)
end

# defined at cfc
@inline function diffusive_flux_y(i, j, k, grid,
                                  closure::Union{ISSD, ISSDVector}, diffusivity_fields, ::Val{tracer_index},
                                  velocities, tracers, clock, buoyancy) where tracer_index

    c = tracers[tracer_index]
    closure = getclosure(i, j, closure)

    κ_skew = get_tracer_κ(closure.κ_skew, tracer_index)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, tracer_index)

    κ_skewᶜᶠᶜ = κᶜᶠᶜ(i, j, k, grid, clock, issd_coefficient_loc, κ_skew)
    κ_symmetricᶜᶠᶜ = κᶜᶠᶜ(i, j, k, grid, clock, issd_coefficient_loc, κ_symmetric)

    ∂y_c = ∂yᶜᶠᶜ(i, j, k, grid, c)

    # Average... of... the gradient!
    ∂x_c = ℑxyᶜᶠᶜ(i, j, k, grid, ∂xᶠᶜᶜ, c)
    ∂z_c = ℑyzᶜᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, c)
    
    # Gradient of the average.
    #∂x_c = ∂xᶜᶠᶜ(i, j, k, grid, ℑxyᶠᶠᶜ, c)
    #∂z_c = ∂zᶜᶠᶜ(i, j, k, grid, ℑyzᶜᶠᶠ, c)

    R₂₁ = zero(grid)
    R₂₂ = one(grid)
    R₂₃ = isopycnal_rotation_tensor_yz_cfc(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    ϵ = taper_factor_ccc(i, j, k, grid, buoyancy, tracers, closure.slope_limiter)

    return - ϵ * (              κ_symmetricᶜᶠᶜ * R₂₁ * ∂x_c +
                                κ_symmetricᶜᶠᶜ * R₂₂ * ∂y_c +
                  (κ_symmetricᶜᶠᶜ - κ_skewᶜᶠᶜ) * R₂₃ * ∂z_c)
end

# defined at ccf
@inline function diffusive_flux_z(i, j, k, grid,
                                  closure::FlavorOfISSD{TD}, diffusivity_fields, ::Val{tracer_index},
                                  velocities, tracers, clock, buoyancy) where {tracer_index, TD}

    c = tracers[tracer_index]
    closure = getclosure(i, j, closure)

    κ_skew = get_tracer_κ(closure.κ_skew, tracer_index)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, tracer_index)

    κ_skewᶜᶜᶠ = κᶜᶜᶠ(i, j, k, grid, clock, issd_coefficient_loc,κ_skew)
    κ_symmetricᶜᶜᶠ = κᶜᶜᶠ(i, j, k, grid, clock, issd_coefficient_loc, κ_symmetric)

    # Average... of... the gradient!
    ∂x_c = ℑxzᶜᶜᶠ(i, j, k, grid, ∂xᶠᶜᶜ, c)
    ∂y_c = ℑyzᶜᶜᶠ(i, j, k, grid, ∂yᶜᶠᶜ, c)

    # Gradient of the average.
    #∂x_c = ∂xᶜᶜᶠ(i, j, k, grid, ℑxzᶠᶜᶠ, c)
    #∂y_c = ∂yᶜᶜᶠ(i, j, k, grid, ℑyzᶜᶠᶠ, c)

    R₃₁ = isopycnal_rotation_tensor_xz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    R₃₂ = isopycnal_rotation_tensor_yz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)

    ϵ = taper_factor_ccc(i, j, k, grid, buoyancy, tracers, closure.slope_limiter)
    κ_symmetric_∂z_c = explicit_κ_∂z_c(i, j, k, grid, TD(), c, κ_symmetricᶜᶜᶠ, closure, buoyancy, tracers)

    return - ϵ * κ_symmetric_∂z_c - ϵ * ((κ_symmetricᶜᶜᶠ + κ_skewᶜᶜᶠ) * R₃₁ * ∂x_c +
                                         (κ_symmetricᶜᶜᶠ + κ_skewᶜᶜᶠ) * R₃₂ * ∂y_c)
end



@inline function explicit_κ_∂z_c(i, j, k, grid, ::ExplicitTimeDiscretization, κ_symmetricᶜᶜᶠ, closure, buoyancy, tracers)
    ∂z_c = ∂zᶜᶜᶠ(i, j, k, grid, c)
    R₃₃ = isopycnal_rotation_tensor_zz_ccf(i, j, k, grid, buoyancy, tracers, closure.isopycnal_tensor)
    return κ_symmetricᶜᶜᶠ * R₃₃ * ∂z_c
end

@inline explicit_κ_∂z_c(i, j, k, grid, ::VerticallyImplicitTimeDiscretization, args...) = zero(grid)

@inline function κzᶜᶜᶠ(i, j, k, grid, closure::FlavorOfISSD, K, ::Val{id}, clock) where id
    closure = getclosure(i, j, closure)
    κ_symmetric = get_tracer_κ(closure.κ_symmetric, id)
    ϵ_R₃₃ = @inbounds K.ϵ_R₃₃[i, j, k] # tapered 33 component of rotation tensor
    return ϵ_R₃₃ * κᶜᶜᶠ(i, j, k, grid, clock, issd_coefficient_loc, κ_symmetric)
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

Base.show(io::IO, closure::ISSD) =
    print(io, "IsopycnalSkewSymmetricDiffusivity: " *
              "(κ_symmetric=$(closure.κ_symmetric), κ_skew=$(closure.κ_skew), " *
              "(isopycnal_tensor=$(closure.isopycnal_tensor), slope_limiter=$(closure.slope_limiter))")
              
