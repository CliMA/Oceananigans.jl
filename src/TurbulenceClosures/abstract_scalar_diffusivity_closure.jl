using Oceananigans.Operators: ℑxyᶠᶠᵃ, ℑxzᶠᵃᶠ, ℑyzᵃᶠᶠ

"""
    abstract type AbstractScalarDiffusivity <: AbstractTurbulenceClosure end

Abstract type for closures with scalar diffusivities.
"""
abstract type AbstractScalarDiffusivity{TD, F, N} <: AbstractTurbulenceClosure{TD, N} end

#####
##### Formulations
#####

abstract type AbstractDiffusivityFormulation end

"""
    struct ThreeDimensionalFormulation end

Specifies a three-dimensionally-isotropic `ScalarDiffusivity`.
"""
struct ThreeDimensionalFormulation <: AbstractDiffusivityFormulation end

"""
    struct HorizontalFormulation end

Specifies a horizontally-isotropic, `VectorInvariant`, `ScalarDiffusivity`.
"""
struct HorizontalFormulation <: AbstractDiffusivityFormulation end

"""
    struct HorizontalDivergenceFormulation end

Specifies viscosity for "divergence damping". Has no effect on tracers.
"""
struct HorizontalDivergenceFormulation <: AbstractDiffusivityFormulation end

"""
    struct VerticalFormulation end

Specifies a `ScalarDiffusivity` acting only in the vertical direction.
"""
struct VerticalFormulation <: AbstractDiffusivityFormulation end

"""
    viscosity(closure, diffusivities)

Returns the scalar viscosity associated with `closure`.
"""
function viscosity end

"""
    diffusivity(closure, tracer_index, diffusivity_fields)

Returns the scalar diffusivity associated with `closure` and `tracer_index`.
"""
function diffusivity end

const c = Center()

# Fallback locations
@inline viscosity_location(::AbstractScalarDiffusivity) = (c, c, c)
@inline diffusivity_location(::AbstractScalarDiffusivity) = (c, c, c)

# For tuples (note that kernel functions are "untupled", so these are for the user API)
viscosity(closure::Tuple, K) = Tuple(viscosity(closure[n], K[n]) for n = 1:length(closure))
diffusivity(closure::Tuple, K, id) = Tuple(diffusivity(closure[n], K[n], id) for n = 1:length(closure))

viscosity(model) = viscosity(model.closure, model.diffusivity_fields)
diffusivity(model, id) = diffusivity(model.closure, model.diffusivity_fields, id)

@inline formulation(::AbstractScalarDiffusivity{TD, F}) where {TD, F} = F()

Base.summary(::VerticalFormulation) = "VerticalFormulation"
Base.summary(::HorizontalFormulation) = "HorizontalFormulation"
Base.summary(::ThreeDimensionalFormulation) = "ThreeDimensionalFormulation"

#####
##### Coefficient extractors
#####

const ASD = AbstractScalarDiffusivity
const AID = AbstractScalarDiffusivity{<:Any, <:ThreeDimensionalFormulation}
const AHD = AbstractScalarDiffusivity{<:Any, <:HorizontalFormulation}
const ADD = AbstractScalarDiffusivity{<:Any, <:HorizontalDivergenceFormulation}
const AVD = AbstractScalarDiffusivity{<:Any, <:VerticalFormulation}

@inline νᶜᶜᶜ(i, j, k, grid, closure::ASD, K, clk, fields) = νᶜᶜᶜ(i, j, k, grid, viscosity_location(closure), viscosity(closure, K), clk, fields)
@inline νᶠᶠᶜ(i, j, k, grid, closure::ASD, K, clk, fields) = νᶠᶠᶜ(i, j, k, grid, viscosity_location(closure), viscosity(closure, K), clk, fields)
@inline νᶠᶜᶠ(i, j, k, grid, closure::ASD, K, clk, fields) = νᶠᶜᶠ(i, j, k, grid, viscosity_location(closure), viscosity(closure, K), clk, fields)
@inline νᶜᶠᶠ(i, j, k, grid, closure::ASD, K, clk, fields) = νᶜᶠᶠ(i, j, k, grid, viscosity_location(closure), viscosity(closure, K), clk, fields)

@inline κᶜᶜᶜ(i, j, k, grid, closure::ASD, K, id, clk, fields) = κᶜᶜᶜ(i, j, k, grid, diffusivity_location(closure), diffusivity(closure, K, id), clk, fields)
@inline κᶠᶜᶜ(i, j, k, grid, closure::ASD, K, id, clk, fields) = κᶠᶜᶜ(i, j, k, grid, diffusivity_location(closure), diffusivity(closure, K, id), clk, fields)
@inline κᶜᶠᶜ(i, j, k, grid, closure::ASD, K, id, clk, fields) = κᶜᶠᶜ(i, j, k, grid, diffusivity_location(closure), diffusivity(closure, K, id), clk, fields)
@inline κᶜᶜᶠ(i, j, k, grid, closure::ASD, K, id, clk, fields) = κᶜᶜᶠ(i, j, k, grid, diffusivity_location(closure), diffusivity(closure, K, id), clk, fields)
@inline κᶠᶜᶠ(i, j, k, grid, closure::ASD, K, id, clk, fields) = κᶠᶜᶠ(i, j, k, grid, diffusivity_location(closure), diffusivity(closure, K, id), clk, fields)
@inline κᶜᶠᶠ(i, j, k, grid, closure::ASD, K, id, clk, fields) = κᶜᶠᶠ(i, j, k, grid, diffusivity_location(closure), diffusivity(closure, K, id), clk, fields)

# Vertical and horizontal diffusivity
@inline νzᶜᶜᶜ(i, j, k, grid, closure::ASD, K, clk, fields) = νᶜᶜᶜ(i, j, k, grid, closure, K, clk, fields)
@inline νzᶠᶠᶜ(i, j, k, grid, closure::ASD, K, clk, fields) = νᶠᶠᶜ(i, j, k, grid, closure, K, clk, fields)
@inline νzᶠᶜᶠ(i, j, k, grid, closure::ASD, K, clk, fields) = νᶠᶜᶠ(i, j, k, grid, closure, K, clk, fields)
@inline νzᶜᶠᶠ(i, j, k, grid, closure::ASD, K, clk, fields) = νᶜᶠᶠ(i, j, k, grid, closure, K, clk, fields)

@inline νzᶜᶜᶜ(i, j, k, grid, closure::ASD, K, ::Nothing, clk, fields) = νzᶜᶜᶜ(i, j, k, grid, closure, K, clk, fields)
@inline νzᶠᶠᶜ(i, j, k, grid, closure::ASD, K, ::Nothing, clk, fields) = νzᶠᶠᶜ(i, j, k, grid, closure, K, clk, fields)
@inline νzᶠᶜᶠ(i, j, k, grid, closure::ASD, K, ::Nothing, clk, fields) = νzᶠᶜᶠ(i, j, k, grid, closure, K, clk, fields)
@inline νzᶜᶠᶠ(i, j, k, grid, closure::ASD, K, ::Nothing, clk, fields) = νzᶜᶠᶠ(i, j, k, grid, closure, K, clk, fields)

@inline κzᶠᶜᶜ(i, j, k, grid, closure::ASD, K, id, clk, fields) = κᶠᶜᶜ(i, j, k, grid, closure, K, id, clk, fields)
@inline κzᶜᶠᶜ(i, j, k, grid, closure::ASD, K, id, clk, fields) = κᶜᶠᶜ(i, j, k, grid, closure, K, id, clk, fields)
@inline κzᶜᶜᶠ(i, j, k, grid, closure::ASD, K, id, clk, fields) = κᶜᶜᶠ(i, j, k, grid, closure, K, id, clk, fields)

@inline νhᶜᶜᶜ(i, j, k, grid, closure::ASD, K, clk, fields) = νᶜᶜᶜ(i, j, k, grid, closure, K, clk, fields)
@inline νhᶠᶠᶜ(i, j, k, grid, closure::ASD, K, clk, fields) = νᶠᶠᶜ(i, j, k, grid, closure, K, clk, fields)
@inline νhᶠᶜᶠ(i, j, k, grid, closure::ASD, K, clk, fields) = νᶠᶜᶠ(i, j, k, grid, closure, K, clk, fields)
@inline νhᶜᶠᶠ(i, j, k, grid, closure::ASD, K, clk, fields) = νᶜᶠᶠ(i, j, k, grid, closure, K, clk, fields)

@inline κhᶠᶜᶜ(i, j, k, grid, closure::ASD, K, id, clk, fields) = κᶠᶜᶜ(i, j, k, grid, closure, K, id, clk, fields)
@inline κhᶜᶠᶜ(i, j, k, grid, closure::ASD, K, id, clk, fields) = κᶜᶠᶜ(i, j, k, grid, closure, K, id, clk, fields)
@inline κhᶜᶜᶠ(i, j, k, grid, closure::ASD, K, id, clk, fields) = κᶜᶜᶠ(i, j, k, grid, closure, K, id, clk, fields)

for (dir, Clo) in zip((:h, :z), (:AVD, :AHD))
    for code in (:ᶜᶜᶜ, :ᶠᶠᶜ, :ᶠᶜᶠ, :ᶜᶠᶠ)
        ν = Symbol(:ν, dir, code)
        @eval begin
            @inline $ν(i, j, k, grid, closure::$Clo, K, clock, args...) = zero(grid)
        end
    end

    for code in (:ᶠᶜᶜ, :ᶜᶠᶜ, :ᶜᶜᶠ)
        κ = Symbol(:κ, dir, code)
        @eval begin
            @inline $κ(i, j, k, grid, closure::$Clo, K, id, clock, args...) = zero(grid)
        end
    end
end

const F = Face
const C = Center

@inline z_diffusivity(i, j, k, grid, ::F, ::C, ::C, closure::ASD, K, id, clk, fields) = κzᶠᶜᶜ(i, j, k, grid, closure, K, id, clk, fields)
@inline z_diffusivity(i, j, k, grid, ::C, ::F, ::C, closure::ASD, K, id, clk, fields) = κzᶜᶠᶜ(i, j, k, grid, closure, K, id, clk, fields)
@inline z_diffusivity(i, j, k, grid, ::C, ::C, ::F, closure::ASD, K, id, clk, fields) = κzᶜᶜᶠ(i, j, k, grid, closure, K, id, clk, fields)

@inline h_diffusivity(i, j, k, grid, ::F, ::C, ::C, closure::ASD, K, id, clk, fields) = κhᶠᶜᶜ(i, j, k, grid, closure, K, id, clk, fields)
@inline h_diffusivity(i, j, k, grid, ::C, ::F, ::C, closure::ASD, K, id, clk, fields) = κhᶜᶠᶜ(i, j, k, grid, closure, K, id, clk, fields)
@inline h_diffusivity(i, j, k, grid, ::C, ::C, ::F, closure::ASD, K, id, clk, fields) = κhᶜᶜᶠ(i, j, k, grid, closure, K, id, clk, fields)

# "diffusivity" with "nothing" index => viscosity of course
@inline z_diffusivity(i, j, k, grid, ::C, ::C, ::C, closure::ASD, K, ::Nothing, clk, fields) = νzᶜᶜᶜ(i, j, k, grid, closure, K, clk, fields)
@inline z_diffusivity(i, j, k, grid, ::F, ::F, ::C, closure::ASD, K, ::Nothing, clk, fields) = νzᶠᶠᶜ(i, j, k, grid, closure, K, clk, fields)
@inline z_diffusivity(i, j, k, grid, ::F, ::C, ::F, closure::ASD, K, ::Nothing, clk, fields) = νzᶠᶜᶠ(i, j, k, grid, closure, K, clk, fields)
@inline z_diffusivity(i, j, k, grid, ::C, ::F, ::F, closure::ASD, K, ::Nothing, clk, fields) = νzᶜᶠᶠ(i, j, k, grid, closure, K, clk, fields)

@inline h_diffusivity(i, j, k, grid, ::C, ::C, ::C, closure::ASD, K, ::Nothing, clk, fields) = νhᶜᶜᶜ(i, j, k, grid, closure, K, clk, fields)
@inline h_diffusivity(i, j, k, grid, ::F, ::F, ::C, closure::ASD, K, ::Nothing, clk, fields) = νhᶠᶠᶜ(i, j, k, grid, closure, K, clk, fields)
@inline h_diffusivity(i, j, k, grid, ::F, ::C, ::F, closure::ASD, K, ::Nothing, clk, fields) = νhᶠᶜᶠ(i, j, k, grid, closure, K, clk, fields)
@inline h_diffusivity(i, j, k, grid, ::C, ::F, ::F, closure::ASD, K, ::Nothing, clk, fields) = νhᶜᶠᶠ(i, j, k, grid, closure, K, clk, fields)

# Horizontal viscous fluxes for isotropic diffusivities
@inline ν_σᶜᶜᶜ(i, j, k, grid, closure, K, clock, fields, σᶜᶜᶜ, args...) = νᶜᶜᶜ(i, j, k, grid, closure, K, clock, fields) * σᶜᶜᶜ(i, j, k, grid, args...)
@inline ν_σᶠᶠᶜ(i, j, k, grid, closure, K, clock, fields, σᶠᶠᶜ, args...) = νᶠᶠᶜ(i, j, k, grid, closure, K, clock, fields) * σᶠᶠᶜ(i, j, k, grid, args...)
@inline ν_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, fields, σᶠᶜᶠ, args...) = νᶠᶜᶠ(i, j, k, grid, closure, K, clock, fields) * σᶠᶜᶠ(i, j, k, grid, args...)
@inline ν_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, fields, σᶜᶠᶠ, args...) = νᶜᶠᶠ(i, j, k, grid, closure, K, clock, fields) * σᶜᶠᶠ(i, j, k, grid, args...)

@inline viscous_flux_ux(i, j, k, grid, clo::AID, K, clk, fields, b) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, Σ₁₁, fields.u, fields.v, fields.w)
@inline viscous_flux_vx(i, j, k, grid, clo::AID, K, clk, fields, b) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, clo, K, clk, fields, Σ₂₁, fields.u, fields.v, fields.w)
@inline viscous_flux_wx(i, j, k, grid, clo::AID, K, clk, fields, b) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, fields, Σ₃₁, fields.u, fields.v, fields.w)
@inline viscous_flux_uy(i, j, k, grid, clo::AID, K, clk, fields, b) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, clo, K, clk, fields, Σ₁₂, fields.u, fields.v, fields.w)
@inline viscous_flux_vy(i, j, k, grid, clo::AID, K, clk, fields, b) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, Σ₂₂, fields.u, fields.v, fields.w)
@inline viscous_flux_wy(i, j, k, grid, clo::AID, K, clk, fields, b) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, fields, Σ₃₂, fields.u, fields.v, fields.w)

# Vertical viscous fluxes for isotropic diffusivities
@inline viscous_flux_uz(i, j, k, grid, clo::AID, K, clk, fields, b) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, fields, Σ₁₃, fields.u, fields.v, fields.w)
@inline viscous_flux_vz(i, j, k, grid, clo::AID, K, clk, fields, b) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, fields, Σ₂₃, fields.u, fields.v, fields.w)
@inline viscous_flux_wz(i, j, k, grid, clo::AID, K, clk, fields, b) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, Σ₃₃, fields.u, fields.v, fields.w)

# Horizontal viscous fluxes for horizontal diffusivities
@inline νh_δᶜᶜᶜ(i, j, k, grid, closure, K, clock, fields, u, v) = νhᶜᶜᶜ(i, j, k, grid, closure, K, clock, fields) * div_xyᶜᶜᶜ(i, j, k, grid, u, v)
@inline νh_ζᶠᶠᶜ(i, j, k, grid, closure, K, clock, fields, u, v) = νhᶠᶠᶜ(i, j, k, grid, closure, K, clock, fields) * ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
@inline νh_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, fields, σᶠᶜᶠ, args...) = νhᶠᶜᶠ(i, j, k, grid, closure, K, clock, fields) * σᶠᶜᶠ(i, j, k, grid, args...)
@inline νh_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, fields, σᶜᶠᶠ, args...) = νhᶜᶠᶠ(i, j, k, grid, closure, K, clock, fields) * σᶜᶠᶠ(i, j, k, grid, args...)

@inline viscous_flux_ux(i, j, k, grid, clo::AHD, K, clk, fields, b) = - νh_δᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, fields.u, fields.v)
@inline viscous_flux_vx(i, j, k, grid, clo::AHD, K, clk, fields, b) = - νh_ζᶠᶠᶜ(i, j, k, grid, clo, K, clk, fields, fields.u, fields.v)
@inline viscous_flux_uy(i, j, k, grid, clo::AHD, K, clk, fields, b) = + νh_ζᶠᶠᶜ(i, j, k, grid, clo, K, clk, fields, fields.u, fields.v)
@inline viscous_flux_vy(i, j, k, grid, clo::AHD, K, clk, fields, b) = - νh_δᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, fields.u, fields.v)
@inline viscous_flux_wx(i, j, k, grid, clo::AHD, K, clk, fields, b) = - νh_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, fields, ∂xᶠᶜᶠ, fields.w)
@inline viscous_flux_wy(i, j, k, grid, clo::AHD, K, clk, fields, b) = - νh_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, fields, ∂yᶜᶠᶠ, fields.w)

# "Divergence damping"
@inline viscous_flux_ux(i, j, k, grid, clo::ADD, K, clk, fields, b) = - νh_δᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, fields.u, fields.v)
@inline viscous_flux_vy(i, j, k, grid, clo::ADD, K, clk, fields, b) = - νh_δᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, fields.u, fields.v)

# Vertical viscous fluxes for vertical diffusivities
@inline νz_σᶜᶜᶜ(i, j, k, grid, closure, K, clock, fields, σᶜᶜᶜ, args...) = νzᶜᶜᶜ(i, j, k, grid, closure, K, clock, fields) * σᶜᶜᶜ(i, j, k, grid, args...)
@inline νz_σᶠᶠᶜ(i, j, k, grid, closure, K, clock, fields, σᶠᶠᶜ, args...) = νzᶠᶠᶜ(i, j, k, grid, closure, K, clock, fields) * σᶠᶠᶜ(i, j, k, grid, args...)
@inline νz_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, fields, σᶠᶜᶠ, args...) = νzᶠᶜᶠ(i, j, k, grid, closure, K, clock, fields) * σᶠᶜᶠ(i, j, k, grid, args...)
@inline νz_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, fields, σᶜᶠᶠ, args...) = νzᶜᶠᶠ(i, j, k, grid, closure, K, clock, fields) * σᶜᶠᶠ(i, j, k, grid, args...)

@inline viscous_flux_uz(i, j, k, grid, clo::AVD, K, clk, fields, b) = - νz_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, fields, ∂zᶠᶜᶠ, fields.u)
@inline viscous_flux_vz(i, j, k, grid, clo::AVD, K, clk, fields, b) = - νz_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, fields, ∂zᶜᶠᶠ, fields.v)
@inline viscous_flux_wz(i, j, k, grid, clo::AVD, K, clk, fields, b) = - νz_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, fields, ∂zᶜᶜᶜ, fields.w)

#####
##### Diffusive fluxes
#####

const AIDorAHD = Union{AID, AHD}
const AIDorAVD = Union{AID, AVD}

@inline diffusive_flux_x(i, j, k, grid, cl::AIDorAHD, K, id, c, clk, fields, b) = - κhᶠᶜᶜ(i, j, k, grid, cl, K, id, clk, fields) * ∂xᶠᶜᶜ(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, cl::AIDorAHD, K, id, c, clk, fields, b) = - κhᶜᶠᶜ(i, j, k, grid, cl, K, id, clk, fields) * ∂yᶜᶠᶜ(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, cl::AIDorAVD, K, id, c, clk, fields, b) = - κzᶜᶜᶠ(i, j, k, grid, cl, K, id, clk, fields) * ∂zᶜᶜᶠ(i, j, k, grid, c)

#####
##### Support for VerticallyImplicit
#####

const VITD = VerticallyImplicitTimeDiscretization

@inline ivd_viscous_flux_uz(i, j, k, grid, closure::AID, K, clock, fields, b) = - ν_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, fields, ∂xᶠᶜᶠ, fields.w)
@inline ivd_viscous_flux_vz(i, j, k, grid, closure::AID, K, clock, fields, b) = - ν_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, fields, ∂yᶜᶠᶠ, fields.w)
@inline ivd_viscous_flux_uz(i, j, k, grid, closure::AVD, K, clock, fields, b) = zero(grid)
@inline ivd_viscous_flux_vz(i, j, k, grid, closure::AVD, K, clock, fields, b) = zero(grid)

# General functions (eg for vertically periodic)
@inline viscous_flux_uz(i, j, k, grid,  ::VITD, closure::AIDorAVD, K, clock, fields, b)        = ivd_viscous_flux_uz(i, j, k, grid, closure, K, clock, fields, b)
@inline viscous_flux_vz(i, j, k, grid,  ::VITD, closure::AIDorAVD, K, clock, fields, b)        = ivd_viscous_flux_vz(i, j, k, grid, closure, K, clock, fields, b)
@inline viscous_flux_wz(i, j, k, grid,  ::VITD, closure::AIDorAVD, K, clock, fields, b)        = zero(grid)
@inline diffusive_flux_z(i, j, k, grid, ::VITD, closure::AIDorAVD, K, id, c, clock, fields, b) = zero(grid)

# Vertically bounded grids
#
# For vertically bounded grids, we calculate _explicit_ fluxes on the boundaries,
# and elide the implicit vertical flux component everywhere else. This is consistent
# with the formulation of the tridiagonal solver, which requires explicit treatment
# of boundary contributions (eg boundary contributions must be moved to the right
# hand side of the tridiagonal system).

@inline function viscous_flux_uz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, K, clk, fields, b)
    return ifelse((k == 1) | (k == grid.Nz+1),
                  viscous_flux_uz(i, j, k, grid, ExplicitTimeDiscretization(), closure, K, clk, fields, b),
                  ivd_viscous_flux_uz(i, j, k, grid, closure, K, clk, fields, b))
end

@inline function viscous_flux_vz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, K, clk, fields, b)
    return ifelse((k == 1) | (k == grid.Nz+1),
                  viscous_flux_vz(i, j, k, grid, ExplicitTimeDiscretization(), closure, K, clk, fields, b),
                  ivd_viscous_flux_vz(i, j, k, grid, closure, K, clk, fields, b))
end

@inline function viscous_flux_wz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, K, clk, fields, b)
    return ifelse((k == 1) | (k == grid.Nz+1),
                  viscous_flux_wz(i, j, k, grid, ExplicitTimeDiscretization(), closure, K, clk, fields, b),
                  zero(grid))
end

@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, K, id, c, clk, fields, b) 
    return ifelse((k == 1) | (k == grid.Nz+1),
                  diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), closure, K, id, c, clk, fields, b),
                  zero(grid))
end

#####
##### Products of viscosity and stress, divergence, vorticity
#####


@inline κ_σᶠᶜᶜ(i, j, k, grid, closure, K, id, clock, fields, σᶠᶜᶜ, args...) = κᶠᶜᶜ(i, j, k, grid, closure, K, id, clock, fields) * σᶠᶜᶜ(i, j, k, grid, args...)
@inline κ_σᶜᶠᶜ(i, j, k, grid, closure, K, id, clock, fields, σᶜᶠᶜ, args...) = κᶜᶠᶜ(i, j, k, grid, closure, K, id, clock, fields) * σᶜᶠᶜ(i, j, k, grid, args...)
@inline κ_σᶜᶜᶠ(i, j, k, grid, closure, K, id, clock, fields, σᶜᶜᶠ, args...) = κᶜᶜᶠ(i, j, k, grid, closure, K, id, clock, fields) * σᶜᶜᶠ(i, j, k, grid, args...)
@inline κ_σᶠᶜᶠ(i, j, k, grid, closure, K, id, clock, fields, σᶠᶜᶠ, args...) = κᶠᶜᶠ(i, j, k, grid, closure, K, id, clock, fields) * σᶠᶜᶠ(i, j, k, grid, args...)
@inline κ_σᶜᶠᶠ(i, j, k, grid, closure, K, id, clock, fields, σᶜᶠᶠ, args...) = κᶜᶠᶠ(i, j, k, grid, closure, K, id, clock, fields) * σᶜᶠᶠ(i, j, k, grid, args...)

#####
##### Viscosity "extractors"
#####

# Number

@inline νᶜᶜᶜ(i, j, k, grid, loc, ν::Number, clk, fields) = ν
@inline νᶠᶜᶠ(i, j, k, grid, loc, ν::Number, clk, fields) = ν
@inline νᶜᶠᶠ(i, j, k, grid, loc, ν::Number, clk, fields) = ν
@inline νᶠᶠᶜ(i, j, k, grid, loc, ν::Number, clk, fields) = ν

@inline κᶠᶜᶜ(i, j, k, grid, loc, κ::Number, clk, fields) = κ
@inline κᶜᶠᶜ(i, j, k, grid, loc, κ::Number, clk, fields) = κ
@inline κᶜᶜᶠ(i, j, k, grid, loc, κ::Number, clk, fields) = κ
@inline κᶠᶜᶠ(i, j, k, grid, loc, κ::Number, clk, fields) = κ
@inline κᶜᶠᶠ(i, j, k, grid, loc, κ::Number, clk, fields) = κ


# Array / Field at `Center, Center, Center`
const Lᶜᶜᶜ = Tuple{Center, Center, Center}
@inline νᶜᶜᶜ(i, j, k, grid, ::Lᶜᶜᶜ, ν::AbstractArray, clk, fields) = @inbounds ν[i, j, k]
@inline νᶠᶜᶠ(i, j, k, grid, ::Lᶜᶜᶜ, ν::AbstractArray, clk, fields) = ℑxzᶠᵃᶠ(i, j, k, grid, ν)
@inline νᶜᶠᶠ(i, j, k, grid, ::Lᶜᶜᶜ, ν::AbstractArray, clk, fields) = ℑyzᵃᶠᶠ(i, j, k, grid, ν)
@inline νᶠᶠᶜ(i, j, k, grid, ::Lᶜᶜᶜ, ν::AbstractArray, clk, fields) = ℑxyᶠᶠᵃ(i, j, k, grid, ν)

@inline κᶠᶜᶜ(i, j, k, grid, ::Lᶜᶜᶜ, κ::AbstractArray, clk, fields) = ℑxᶠᵃᵃ(i, j, k, grid, κ)
@inline κᶜᶠᶜ(i, j, k, grid, ::Lᶜᶜᶜ, κ::AbstractArray, clk, fields) = ℑyᵃᶠᵃ(i, j, k, grid, κ)
@inline κᶜᶜᶠ(i, j, k, grid, ::Lᶜᶜᶜ, κ::AbstractArray, clk, fields) = ℑzᵃᵃᶠ(i, j, k, grid, κ)
@inline κᶠᶜᶠ(i, j, k, grid, ::Lᶜᶜᶜ, κ::AbstractArray, clk, fields) = ℑxzᶠᵃᶠ(i, j, k, grid, κ)
@inline κᶜᶠᶠ(i, j, k, grid, ::Lᶜᶜᶜ, κ::AbstractArray, clk, fields) = ℑyzᵃᶠᶠ(i, j, k, grid, κ)

# Array / Field at `Center, Center, Face`
const Lᶜᶜᶠ = Tuple{Center, Center, Face}
@inline νᶜᶜᶜ(i, j, k, grid, ::Lᶜᶜᶠ, ν::AbstractArray, clk, fields) = ℑzᵃᵃᶜ(i, j, k, grid, ν)
@inline νᶠᶜᶠ(i, j, k, grid, ::Lᶜᶜᶠ, ν::AbstractArray, clk, fields) = ℑxᶠᵃᵃ(i, j, k, grid, ν)
@inline νᶜᶠᶠ(i, j, k, grid, ::Lᶜᶜᶠ, ν::AbstractArray, clk, fields) = ℑyᵃᶠᵃ(i, j, k, grid, ν)
@inline νᶠᶠᶜ(i, j, k, grid, ::Lᶜᶜᶠ, ν::AbstractArray, clk, fields) = ℑxyzᶠᶠᶜ(i, j, k, grid, ν)

@inline κᶠᶜᶜ(i, j, k, grid, ::Lᶜᶜᶠ, κ::AbstractArray, clk, fields) = ℑxzᶠᵃᶠ(i, j, k, grid, κ)
@inline κᶜᶠᶜ(i, j, k, grid, ::Lᶜᶜᶠ, κ::AbstractArray, clk, fields) = ℑyzᵃᶠᶠ(i, j, k, grid, κ)
@inline κᶜᶜᶠ(i, j, k, grid, ::Lᶜᶜᶠ, κ::AbstractArray, clk, fields) = @inbounds κ[i, j, k]
@inline κᶠᶜᶠ(i, j, k, grid, ::Lᶜᶜᶠ, κ::AbstractArray, clk, fields) = ℑxᶠᵃᵃ(i, j, k, grid, κ)
@inline κᶜᶠᶠ(i, j, k, grid, ::Lᶜᶜᶠ, κ::AbstractArray, clk, fields) = ℑyᵃᶠᵃ(i, j, k, grid, κ)

# Function

const c = Center()
const f = Face()

@inline νᶜᶜᶜ(i, j, k, grid, loc, ν::Function, clock, fields) = ν(node(i, j, k, grid, c, c, c)..., clock.time)
@inline νᶠᶜᶠ(i, j, k, grid, loc, ν::Function, clock, fields) = ν(node(i, j, k, grid, f, c, f)..., clock.time)
@inline νᶜᶠᶠ(i, j, k, grid, loc, ν::Function, clock, fields) = ν(node(i, j, k, grid, c, f, f)..., clock.time)
@inline νᶠᶠᶜ(i, j, k, grid, loc, ν::Function, clock, fields) = ν(node(i, j, k, grid, f, f, c)..., clock.time)

@inline κᶜᶜᶜ(i, j, k, grid, loc, κ::Function, clock, fields) = κ(node(i, j, k, grid, c, c, c)..., clock.time)
@inline κᶠᶜᶜ(i, j, k, grid, loc, κ::Function, clock, fields) = κ(node(i, j, k, grid, f, c, c)..., clock.time)
@inline κᶜᶠᶜ(i, j, k, grid, loc, κ::Function, clock, fields) = κ(node(i, j, k, grid, c, f, c)..., clock.time)
@inline κᶜᶜᶠ(i, j, k, grid, loc, κ::Function, clock, fields) = κ(node(i, j, k, grid, c, c, f)..., clock.time)
@inline κᶠᶜᶠ(i, j, k, grid, loc, κ::Function, clock, fields) = κ(node(i, j, k, grid, f, c, f)..., clock.time)
@inline κᶜᶠᶠ(i, j, k, grid, loc, κ::Function, clock, fields) = κ(node(i, j, k, grid, c, f, f)..., clock.time)

# "DiscreteDiffusionFunction"
@inline νᶜᶜᶜ(i, j, k, grid, loc, ν::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(ν, i, j, k, grid, (c, c, c), clock, fields)
@inline νᶠᶜᶠ(i, j, k, grid, loc, ν::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(ν, i, j, k, grid, (f, c, f), clock, fields)
@inline νᶜᶠᶠ(i, j, k, grid, loc, ν::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(ν, i, j, k, grid, (c, f, f), clock, fields)
@inline νᶠᶠᶜ(i, j, k, grid, loc, ν::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(ν, i, j, k, grid, (f, f, c), clock, fields)

@inline κᶜᶜᶜ(i, j, k, grid, loc, κ::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(κ, i, j, k, grid, (c, c, c), clock, fields)
@inline κᶠᶜᶜ(i, j, k, grid, loc, κ::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(κ, i, j, k, grid, (f, c, c), clock, fields)
@inline κᶜᶠᶜ(i, j, k, grid, loc, κ::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(κ, i, j, k, grid, (c, f, c), clock, fields)
@inline κᶜᶜᶠ(i, j, k, grid, loc, κ::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(κ, i, j, k, grid, (c, c, f), clock, fields)
@inline κᶠᶜᶠ(i, j, k, grid, loc, κ::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(κ, i, j, k, grid, (f, c, f), clock, fields)
@inline κᶜᶠᶠ(i, j, k, grid, loc, κ::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(κ, i, j, k, grid, (c, f, f), clock, fields)
