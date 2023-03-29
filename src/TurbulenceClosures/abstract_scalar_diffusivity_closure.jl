using Oceananigans.Operators: ℑxyᶠᶠᵃ, ℑxzᶠᵃᶠ, ℑyzᵃᶠᶠ

"""
    abstract type AbstractScalarDiffusivity <: AbstractTurbulenceClosure end

Abstract type for closures with scalar diffusivities.
"""
abstract type AbstractScalarDiffusivity{TD, F} <: AbstractTurbulenceClosure{TD} end

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

# Fallback locations
@inline viscosity_location(::AbstractScalarDiffusivity) = (Center(), Center(), Center())
@inline diffusivity_location(::AbstractScalarDiffusivity) = (Center(), Center(), Center())

# For tuples (note that kernel functions are "untupled", so these are for the user API)
viscosity(closure::Tuple, K) = Tuple(viscosity(closure[n], K[n]) for n = 1:length(closure))
diffusivity(closure::Tuple, K, id) = Tuple(diffusivity(closure[n], K[n], id) for n = 1:length(closure))

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

@inline νᶜᶜᶜ(i, j, k, grid, closure::ASD, K, args...)     = νᶜᶜᶜ(i, j, k, grid, viscosity_location(closure), viscosity(closure, K), args...) 
@inline νᶠᶠᶜ(i, j, k, grid, closure::ASD, K, args...)     = νᶠᶠᶜ(i, j, k, grid, viscosity_location(closure), viscosity(closure, K), args...)
@inline νᶠᶜᶠ(i, j, k, grid, closure::ASD, K, args...)     = νᶠᶜᶠ(i, j, k, grid, viscosity_location(closure), viscosity(closure, K), args...)
@inline νᶜᶠᶠ(i, j, k, grid, closure::ASD, K, args...)     = νᶜᶠᶠ(i, j, k, grid, viscosity_location(closure), viscosity(closure, K), args...)

@inline κᶠᶜᶜ(i, j, k, grid, closure::ASD, K, id, args...) = κᶠᶜᶜ(i, j, k, grid, diffusivity_location(closure), diffusivity(closure, K, id), args...)
@inline κᶜᶠᶜ(i, j, k, grid, closure::ASD, K, id, args...) = κᶜᶠᶜ(i, j, k, grid, diffusivity_location(closure), diffusivity(closure, K, id), args...)
@inline κᶜᶜᶠ(i, j, k, grid, closure::ASD, K, id, args...) = κᶜᶜᶠ(i, j, k, grid, diffusivity_location(closure), diffusivity(closure, K, id), args...)

# Vertical and horizontal diffusivity
@inline νzᶜᶜᶜ(i, j, k, grid, closure::ASD, K, args...)     = νᶜᶜᶜ(i, j, k, grid, closure, K, args...) 
@inline νzᶠᶠᶜ(i, j, k, grid, closure::ASD, K, args...)     = νᶠᶠᶜ(i, j, k, grid, closure, K, args...) 
@inline νzᶠᶜᶠ(i, j, k, grid, closure::ASD, K, args...)     = νᶠᶜᶠ(i, j, k, grid, closure, K, args...) 
@inline νzᶜᶠᶠ(i, j, k, grid, closure::ASD, K, args...)     = νᶜᶠᶠ(i, j, k, grid, closure, K, args...) 
@inline νzᶠᶜᶠ(i, j, k, grid, closure::ASD, K, ::Nothing, args...) = νzᶠᶜᶠ(i, j, k, grid, closure, K, args...)
@inline νzᶜᶠᶠ(i, j, k, grid, closure::ASD, K, ::Nothing, args...) = νzᶜᶠᶠ(i, j, k, grid, closure, K, args...)

@inline κzᶠᶜᶜ(i, j, k, grid, closure::ASD, K, id, args...) = κᶠᶜᶜ(i, j, k, grid, closure, K, id, args...)
@inline κzᶜᶠᶜ(i, j, k, grid, closure::ASD, K, id, args...) = κᶜᶠᶜ(i, j, k, grid, closure, K, id, args...)
@inline κzᶜᶜᶠ(i, j, k, grid, closure::ASD, K, id, args...) = κᶜᶜᶠ(i, j, k, grid, closure, K, id, args...)

@inline νhᶜᶜᶜ(i, j, k, grid, closure::ASD, K, args...)     = νᶜᶜᶜ(i, j, k, grid, closure, K, args...) 
@inline νhᶠᶠᶜ(i, j, k, grid, closure::ASD, K, args...)     = νᶠᶠᶜ(i, j, k, grid, closure, K, args...) 
@inline νhᶠᶜᶠ(i, j, k, grid, closure::ASD, K, args...)     = νᶠᶜᶠ(i, j, k, grid, closure, K, args...) 
@inline νhᶜᶠᶠ(i, j, k, grid, closure::ASD, K, args...)     = νᶜᶠᶠ(i, j, k, grid, closure, K, args...) 

@inline κhᶠᶜᶜ(i, j, k, grid, closure::ASD, K, id, args...) = κᶠᶜᶜ(i, j, k, grid, closure, K, id, args...)
@inline κhᶜᶠᶜ(i, j, k, grid, closure::ASD, K, id, args...) = κᶜᶠᶜ(i, j, k, grid, closure, K, id, args...)
@inline κhᶜᶜᶠ(i, j, k, grid, closure::ASD, K, id, args...) = κᶜᶜᶠ(i, j, k, grid, closure, K, id, args...)

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

@inline z_diffusivity(i, j, k, grid, ::F, ::C, ::C, closure::ASD, K, id, args...) = κzᶠᶜᶜ(i, j, k, grid, closure, K, id, args...)
@inline z_diffusivity(i, j, k, grid, ::C, ::F, ::C, closure::ASD, K, id, args...) = κzᶜᶠᶜ(i, j, k, grid, closure, K, id, args...)
@inline z_diffusivity(i, j, k, grid, ::C, ::C, ::F, closure::ASD, K, id, args...) = κzᶜᶜᶠ(i, j, k, grid, closure, K, id, args...)

@inline h_diffusivity(i, j, k, grid, ::F, ::C, ::C, closure::ASD, K, id, args...) = κhᶠᶜᶜ(i, j, k, grid, closure, K, id, args...)
@inline h_diffusivity(i, j, k, grid, ::C, ::F, ::C, closure::ASD, K, id, args...) = κhᶜᶠᶜ(i, j, k, grid, closure, K, id, args...)
@inline h_diffusivity(i, j, k, grid, ::C, ::C, ::F, closure::ASD, K, id, args...) = κhᶜᶜᶠ(i, j, k, grid, closure, K, id, args...)

# "diffusivity" with "nothing" index => viscosity of course
@inline z_diffusivity(i, j, k, grid, ::C, ::C, ::C, closure::ASD, K, ::Nothing, args...) = νzᶜᶜᶜ(i, j, k, grid, closure, K, args...)
@inline z_diffusivity(i, j, k, grid, ::F, ::F, ::C, closure::ASD, K, ::Nothing, args...) = νzᶠᶠᶜ(i, j, k, grid, closure, K, args...)
@inline z_diffusivity(i, j, k, grid, ::F, ::C, ::F, closure::ASD, K, ::Nothing, args...) = νzᶠᶜᶠ(i, j, k, grid, closure, K, args...)
@inline z_diffusivity(i, j, k, grid, ::C, ::F, ::F, closure::ASD, K, ::Nothing, args...) = νzᶜᶠᶠ(i, j, k, grid, closure, K, args...)

@inline h_diffusivity(i, j, k, grid, ::C, ::C, ::C, closure::ASD, K, ::Nothing, args...) = νhᶜᶜᶜ(i, j, k, grid, closure, K, args...)
@inline h_diffusivity(i, j, k, grid, ::F, ::F, ::C, closure::ASD, K, ::Nothing, args...) = νhᶠᶠᶜ(i, j, k, grid, closure, K, args...)
@inline h_diffusivity(i, j, k, grid, ::F, ::C, ::F, closure::ASD, K, ::Nothing, args...) = νhᶠᶜᶠ(i, j, k, grid, closure, K, args...)
@inline h_diffusivity(i, j, k, grid, ::C, ::F, ::F, closure::ASD, K, ::Nothing, args...) = νhᶜᶠᶠ(i, j, k, grid, closure, K, args...)

#####
##### Stress divergences
#####

#####
##### Fallback: flux = 0
#####

for dir in (:x, :y, :z)
    diffusive_flux = Symbol(:diffusive_flux_, dir)
    viscous_flux_u = Symbol(:viscous_flux_u, dir)
    viscous_flux_v = Symbol(:viscous_flux_v, dir)
    viscous_flux_w = Symbol(:viscous_flux_w, dir)
    @eval begin
        @inline $diffusive_flux(i, j, k, grid, args...) = zero(grid)
        @inline $viscous_flux_u(i, j, k, grid, args...) = zero(grid)
        @inline $viscous_flux_v(i, j, k, grid, args...) = zero(grid)
        @inline $viscous_flux_w(i, j, k, grid, args...) = zero(grid)
    end
end


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

@inline diffusive_flux_x(i, j, k, grid, cl::AIDorAHD, K, ::Val{id}, c, clk, fields, b) where id = - κhᶠᶜᶜ(i, j, k, grid, cl, K, Val(id), clk, fields) * ∂xᶠᶜᶜ(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, cl::AIDorAHD, K, ::Val{id}, c, clk, fields, b) where id = - κhᶜᶠᶜ(i, j, k, grid, cl, K, Val(id), clk, fields) * ∂yᶜᶠᶜ(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, cl::AIDorAVD, K, ::Val{id}, c, clk, fields, b) where id = - κzᶜᶜᶠ(i, j, k, grid, cl, K, Val(id), clk, fields) * ∂zᶜᶜᶠ(i, j, k, grid, c)

#####
##### Support for VerticallyImplicit
#####

const VITD = VerticallyImplicitTimeDiscretization

@inline ivd_viscous_flux_uz(i, j, k, grid, closure::AID, K, clock, fields, b) = - ν_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, fields, ∂xᶠᶜᶠ, fields.w)
@inline ivd_viscous_flux_vz(i, j, k, grid, closure::AID, K, clock, fields, b) = - ν_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, fields, ∂yᶜᶠᶠ, fields.w)
@inline ivd_viscous_flux_uz(i, j, k, grid, closure::AVD, K, clock, fields, b) = zero(grid)
@inline ivd_viscous_flux_vz(i, j, k, grid, closure::AVD, K, clock, fields, b) = zero(grid)

# General functions (eg for vertically periodic)
@inline viscous_flux_uz(i, j, k, grid,  ::VITD, closure::AIDorAVD, args...) = ivd_viscous_flux_uz(i, j, k, grid, closure, args...)
@inline viscous_flux_vz(i, j, k, grid,  ::VITD, closure::AIDorAVD, args...) = ivd_viscous_flux_vz(i, j, k, grid, closure, args...)
@inline viscous_flux_wz(i, j, k, grid,  ::VITD, closure::AIDorAVD, args...) = zero(grid)
@inline diffusive_flux_z(i, j, k, grid, ::VITD, closure::AIDorAVD, args...) = zero(grid)
                  
# Vertically bounded grids
#
# For vertically bounded grids, we calculate _explicit_ fluxes on the boundaries, 
# and elide the implicit vertical flux component everywhere else. This is consistent
# with the formulation of the tridiagonal solver, which requires explicit treatment
# of boundary contributions (eg boundary contributions must be moved to the right
# hand side of the tridiagonal system).

@inline function viscous_flux_uz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, args...)
    return ifelse((k == 1) | (k == grid.Nz+1), 
                  viscous_flux_uz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  ivd_viscous_flux_uz(i, j, k, grid, closure, args...))
end

@inline function viscous_flux_vz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, args...)
    return ifelse((k == 1) | (k == grid.Nz+1), 
                  viscous_flux_vz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  ivd_viscous_flux_vz(i, j, k, grid, closure, args...))
end

@inline function viscous_flux_wz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, args...)
    return ifelse((k == 1) | (k == grid.Nz+1), 
                  viscous_flux_wz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  zero(grid))
end

@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, args...)
    return ifelse((k == 1) | (k == grid.Nz+1), 
                  diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  zero(grid))
end

#####
##### Products of viscosity and stress, divergence, vorticity
#####


@inline κ_σᶠᶜᶜ(i, j, k, grid, closure, K, id, clock, fields, σᶠᶜᶜ, args...) = κᶠᶜᶜ(i, j, k, grid, closure, K, id, clock, fields) * σᶠᶜᶜ(i, j, k, grid, args...)
@inline κ_σᶜᶠᶜ(i, j, k, grid, closure, K, id, clock, fields, σᶜᶠᶜ, args...) = κᶜᶠᶜ(i, j, k, grid, closure, K, id, clock, fields) * σᶜᶠᶜ(i, j, k, grid, args...)
@inline κ_σᶜᶜᶠ(i, j, k, grid, closure, K, id, clock, fields, σᶜᶜᶠ, args...) = κᶜᶜᶠ(i, j, k, grid, closure, K, id, clock, fields) * σᶜᶜᶠ(i, j, k, grid, args...)

#####
##### Viscosity "extractors"
#####

# Number

@inline νᶜᶜᶜ(i, j, k, grid, loc, ν::Number, args...) = ν
@inline νᶠᶜᶠ(i, j, k, grid, loc, ν::Number, args...) = ν
@inline νᶜᶠᶠ(i, j, k, grid, loc, ν::Number, args...) = ν
@inline νᶠᶠᶜ(i, j, k, grid, loc, ν::Number, args...) = ν

@inline κᶠᶜᶜ(i, j, k, grid, loc, κ::Number, args...) = κ
@inline κᶜᶠᶜ(i, j, k, grid, loc, κ::Number, args...) = κ
@inline κᶜᶜᶠ(i, j, k, grid, loc, κ::Number, args...) = κ

# Array / Field at `Center, Center, Center`
const Lᶜᶜᶜ = Tuple{Center, Center, Center}
@inline νᶜᶜᶜ(i, j, k, grid, ::Lᶜᶜᶜ, ν::AbstractArray, args...) = @inbounds ν[i, j, k]
@inline νᶠᶜᶠ(i, j, k, grid, ::Lᶜᶜᶜ, ν::AbstractArray, args...) = ℑxzᶠᵃᶠ(i, j, k, grid, ν)
@inline νᶜᶠᶠ(i, j, k, grid, ::Lᶜᶜᶜ, ν::AbstractArray, args...) = ℑyzᵃᶠᶠ(i, j, k, grid, ν)
@inline νᶠᶠᶜ(i, j, k, grid, ::Lᶜᶜᶜ, ν::AbstractArray, args...) = ℑxyᶠᶠᵃ(i, j, k, grid, ν)
                                        
@inline κᶠᶜᶜ(i, j, k, grid, ::Lᶜᶜᶜ, κ::AbstractArray, args...) = ℑxᶠᵃᵃ(i, j, k, grid, κ)
@inline κᶜᶠᶜ(i, j, k, grid, ::Lᶜᶜᶜ, κ::AbstractArray, args...) = ℑyᵃᶠᵃ(i, j, k, grid, κ)
@inline κᶜᶜᶠ(i, j, k, grid, ::Lᶜᶜᶜ, κ::AbstractArray, args...) = ℑzᵃᵃᶠ(i, j, k, grid, κ)

# Array / Field at `Center, Center, Face`
const Lᶜᶜᶠ = Tuple{Center, Center, Face}
@inline νᶜᶜᶜ(i, j, k, grid, ::Lᶜᶜᶠ, ν::AbstractArray, args...) = ℑzᵃᵃᶜ(i, j, k, grid, ν)
@inline νᶠᶜᶠ(i, j, k, grid, ::Lᶜᶜᶠ, ν::AbstractArray, args...) = ℑxᶠᵃᵃ(i, j, k, grid, ν)
@inline νᶜᶠᶠ(i, j, k, grid, ::Lᶜᶜᶠ, ν::AbstractArray, args...) = ℑyᵃᶠᵃ(i, j, k, grid, ν)
@inline νᶠᶠᶜ(i, j, k, grid, ::Lᶜᶜᶠ, ν::AbstractArray, args...) = ℑxyzᶠᶠᶜ(i, j, k, grid, ν)

@inline κᶠᶜᶜ(i, j, k, grid, ::Lᶜᶜᶠ, κ::AbstractArray, args...) = ℑxzᶠᵃᶠ(i, j, k, grid, κ)
@inline κᶜᶠᶜ(i, j, k, grid, ::Lᶜᶜᶠ, κ::AbstractArray, args...) = ℑyzᵃᶠᶠ(i, j, k, grid, κ)
@inline κᶜᶜᶠ(i, j, k, grid, ::Lᶜᶜᶠ, κ::AbstractArray, args...) = @inbounds κ[i, j, k]

# Function

const c = Center()
const f = Face()

@inline νᶜᶜᶜ(i, j, k, grid, loc, ν::F, clock, args...) where F<:Function = ν(node(i, j, k, grid, c, c, c)..., clock.time)
@inline νᶠᶜᶠ(i, j, k, grid, loc, ν::F, clock, args...) where F<:Function = ν(node(i, j, k, grid, f, c, f)..., clock.time)
@inline νᶜᶠᶠ(i, j, k, grid, loc, ν::F, clock, args...) where F<:Function = ν(node(i, j, k, grid, c, f, f)..., clock.time)
@inline νᶠᶠᶜ(i, j, k, grid, loc, ν::F, clock, args...) where F<:Function = ν(node(i, j, k, grid, f, f, c)..., clock.time)

@inline κᶠᶜᶜ(i, j, k, grid, loc, κ::F, clock, args...) where F<:Function = κ(node(i, j, k, grid, f, c, c)..., clock.time)
@inline κᶜᶠᶜ(i, j, k, grid, loc, κ::F, clock, args...) where F<:Function = κ(node(i, j, k, grid, c, f, c)..., clock.time)
@inline κᶜᶜᶠ(i, j, k, grid, loc, κ::F, clock, args...) where F<:Function = κ(node(i, j, k, grid, c, c, f)..., clock.time)

# "DiscreteDiffusionFunction"
@inline νᶜᶜᶜ(i, j, k, grid, loc, ν::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(ν, i, j, k, grid, (c, c, c), clock, fields)
@inline νᶠᶜᶠ(i, j, k, grid, loc, ν::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(ν, i, j, k, grid, (f, c, f), clock, fields)
@inline νᶜᶠᶠ(i, j, k, grid, loc, ν::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(ν, i, j, k, grid, (c, f, f), clock, fields)
@inline νᶠᶠᶜ(i, j, k, grid, loc, ν::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(ν, i, j, k, grid, (f, f, c), clock, fields)

@inline κᶠᶜᶜ(i, j, k, grid, loc, κ::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(κ, i, j, k, grid, (f, c, c), clock, fields)
@inline κᶜᶠᶜ(i, j, k, grid, loc, κ::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(κ, i, j, k, grid, (c, f, c), clock, fields)
@inline κᶜᶜᶠ(i, j, k, grid, loc, κ::DiscreteDiffusionFunction, clock, fields) = getdiffusivity(κ, i, j, k, grid, (c, c, f), clock, fields)


