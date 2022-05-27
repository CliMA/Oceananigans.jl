"""
    abstract type AbstractScalarDiffusivity <: AbstractTurbulenceClosure end

Abstract type for closures with *isotropic* diffusivities.
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
viscosity(closure::Tuple, K) = sum(Tuple(viscosity(closure[n], K[n]) for n = 1:length(closure)))
diffusivity(closure::Tuple, K, id) = sum(Tuple(diffusivity(closure[n], K[n], id) for n = 1:length(closure)))

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
const AVD = AbstractScalarDiffusivity{<:Any, <:VerticalFormulation}

@inline νᶜᶜᶜ(i, j, k, grid, clo::ASD, K, clk, F)     = νᶜᶜᶜ(i, j, k, grid, clk, viscosity_location(clo), viscosity(clo, K), F) 
@inline νᶠᶠᶜ(i, j, k, grid, clo::ASD, K, clk, F)     = νᶠᶠᶜ(i, j, k, grid, clk, viscosity_location(clo), viscosity(clo, K), F)
@inline νᶠᶜᶠ(i, j, k, grid, clo::ASD, K, clk, F)     = νᶠᶜᶠ(i, j, k, grid, clk, viscosity_location(clo), viscosity(clo, K), F)
@inline νᶜᶠᶠ(i, j, k, grid, clo::ASD, K, clk, F)     = νᶜᶠᶠ(i, j, k, grid, clk, viscosity_location(clo), viscosity(clo, K), F)

@inline κᶠᶜᶜ(i, j, k, grid, clo::ASD, K, id, clk, F) = κᶠᶜᶜ(i, j, k, grid, clk, diffusivity_location(clo), diffusivity(clo, K, id), F)
@inline κᶜᶠᶜ(i, j, k, grid, clo::ASD, K, id, clk, F) = κᶜᶠᶜ(i, j, k, grid, clk, diffusivity_location(clo), diffusivity(clo, K, id), F)
@inline κᶜᶜᶠ(i, j, k, grid, clo::ASD, K, id, clk, F) = κᶜᶜᶠ(i, j, k, grid, clk, diffusivity_location(clo), diffusivity(clo, K, id), F)

# Vertical and horizontal diffusivity
@inline νzᶜᶜᶜ(i, j, k, grid, clo::ASD, K, clock, F)     = νᶜᶜᶜ(i, j, k, grid, clo, K, clock, F) 
@inline νzᶠᶠᶜ(i, j, k, grid, clo::ASD, K, clock, F)     = νᶠᶠᶜ(i, j, k, grid, clo, K, clock, F) 
@inline νzᶠᶜᶠ(i, j, k, grid, clo::ASD, K, clock, F)     = νᶠᶜᶠ(i, j, k, grid, clo, K, clock, F) 
@inline νzᶜᶠᶠ(i, j, k, grid, clo::ASD, K, clock, F)     = νᶜᶠᶠ(i, j, k, grid, clo, K, clock, F) 
@inline κzᶠᶜᶜ(i, j, k, grid, clo::ASD, K, id, clock, F) = κᶠᶜᶜ(i, j, k, grid, clo, K, id, clock, F)
@inline κzᶜᶠᶜ(i, j, k, grid, clo::ASD, K, id, clock, F) = κᶜᶠᶜ(i, j, k, grid, clo, K, id, clock, F)
@inline κzᶜᶜᶠ(i, j, k, grid, clo::ASD, K, id, clock, F) = κᶜᶜᶠ(i, j, k, grid, clo, K, id, clock, F)

@inline νhᶜᶜᶜ(i, j, k, grid, clo::ASD, K, clock, F)     = νᶜᶜᶜ(i, j, k, grid, clo, K, clock, F) 
@inline νhᶠᶠᶜ(i, j, k, grid, clo::ASD, K, clock, F)     = νᶠᶠᶜ(i, j, k, grid, clo, K, clock, F) 
@inline νhᶠᶜᶠ(i, j, k, grid, clo::ASD, K, clock, F)     = νᶠᶜᶠ(i, j, k, grid, clo, K, clock, F) 
@inline νhᶜᶠᶠ(i, j, k, grid, clo::ASD, K, clock, F)     = νᶜᶠᶠ(i, j, k, grid, clo, K, clock, F) 
@inline κhᶠᶜᶜ(i, j, k, grid, clo::ASD, K, id, clock, F) = κᶠᶜᶜ(i, j, k, grid, clo, K, id, clock, F)
@inline κhᶜᶠᶜ(i, j, k, grid, clo::ASD, K, id, clock, F) = κᶜᶠᶜ(i, j, k, grid, clo, K, id, clock, F)
@inline κhᶜᶜᶠ(i, j, k, grid, clo::ASD, K, id, clock, F) = κᶜᶜᶠ(i, j, k, grid, clo, K, id, clock, F)

for (dir, Clo) in zip((:h, :z), (:AVD, :AHD))
    for code in (:ᶜᶜᶜ, :ᶠᶠᶜ, :ᶠᶜᶠ, :ᶜᶠᶠ)
        ν = Symbol(:ν, dir, code)
        @eval begin
            @inline $ν(i, j, k, grid, clo::$Clo, K, clock, F) = zero(eltype(grid))
        end
    end

    for code in (:ᶠᶜᶜ, :ᶜᶠᶜ, :ᶜᶜᶠ)
        κ = Symbol(:κ, dir, code)
        @eval begin
            @inline $κ(i, j, k, grid, clo::$Clo, K, id, clock, F) = zero(eltype(grid))
        end
    end
end

const F = Face
const C = Center

@inline z_diffusivity(i, j, k, grid, ::F, ::C, ::C, clo::ASD, K, id, clock, F) = κzᶠᶜᶜ(i, j, k, grid, clo, K, id, clock, F)
@inline z_diffusivity(i, j, k, grid, ::C, ::F, ::C, clo::ASD, K, id, clock, F) = κzᶜᶠᶜ(i, j, k, grid, clo, K, id, clock, F)
@inline z_diffusivity(i, j, k, grid, ::C, ::C, ::F, clo::ASD, K, id, clock, F) = κzᶜᶜᶠ(i, j, k, grid, clo, K, id, clock, F)

@inline h_diffusivity(i, j, k, grid, ::F, ::C, ::C, clo::ASD, K, id, clock, F) = κhᶠᶜᶜ(i, j, k, grid, clo, K, id, clock, F)
@inline h_diffusivity(i, j, k, grid, ::C, ::F, ::C, clo::ASD, K, id, clock, F) = κhᶜᶠᶜ(i, j, k, grid, clo, K, id, clock, F)
@inline h_diffusivity(i, j, k, grid, ::C, ::C, ::F, clo::ASD, K, id, clock, F) = κhᶜᶜᶠ(i, j, k, grid, clo, K, id, clock, F)

# "diffusivity" with "nothing" index => viscosity of course
@inline z_diffusivity(i, j, k, grid, ::C, ::C, ::C, clo::ASD, K, ::Nothing, clock, F) = νzᶜᶜᶜ(i, j, k, grid, clo, K, clock, F)
@inline z_diffusivity(i, j, k, grid, ::F, ::F, ::C, clo::ASD, K, ::Nothing, clock, F) = νzᶠᶠᶜ(i, j, k, grid, clo, K, clock, F)
@inline z_diffusivity(i, j, k, grid, ::F, ::C, ::F, clo::ASD, K, ::Nothing, clock, F) = νzᶠᶜᶠ(i, j, k, grid, clo, K, clock, F)
@inline z_diffusivity(i, j, k, grid, ::C, ::F, ::F, clo::ASD, K, ::Nothing, clock, F) = νzᶜᶠᶠ(i, j, k, grid, clo, K, clock, F)

@inline h_diffusivity(i, j, k, grid, ::C, ::C, ::C, clo::ASD, K, ::Nothing, clock, F) = νhᶜᶜᶜ(i, j, k, grid, clo, K, clock, F)
@inline h_diffusivity(i, j, k, grid, ::F, ::F, ::C, clo::ASD, K, ::Nothing, clock, F) = νhᶠᶠᶜ(i, j, k, grid, clo, K, clock, F)
@inline h_diffusivity(i, j, k, grid, ::F, ::C, ::F, clo::ASD, K, ::Nothing, clock, F) = νhᶠᶜᶠ(i, j, k, grid, clo, K, clock, F)
@inline h_diffusivity(i, j, k, grid, ::C, ::F, ::F, clo::ASD, K, ::Nothing, clock, F) = νhᶜᶠᶠ(i, j, k, grid, clo, K, clock, F)

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
    
const AID = AbstractScalarDiffusivity{<:Any, <:ThreeDimensionalFormulation}
const AHD = AbstractScalarDiffusivity{<:Any, <:HorizontalFormulation}
const ADD = AbstractScalarDiffusivity{<:Any, <:HorizontalDivergenceFormulation}
const AVD = AbstractScalarDiffusivity{<:Any, <:VerticalFormulation}

@inline viscous_flux_ux(i, j, k, grid, clo::AID, K, clk, F, b) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, F, Σ₁₁, F.u, F.v, F.w)
@inline viscous_flux_vx(i, j, k, grid, clo::AID, K, clk, F, b) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, clo, K, clk, F, Σ₂₁, F.u, F.v, F.w)
@inline viscous_flux_wx(i, j, k, grid, clo::AID, K, clk, F, b) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, F, Σ₃₁, F.u, F.v, F.w)
@inline viscous_flux_uy(i, j, k, grid, clo::AID, K, clk, F, b) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, clo, K, clk, F, Σ₁₂, F.u, F.v, F.w)
@inline viscous_flux_vy(i, j, k, grid, clo::AID, K, clk, F, b) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, F, Σ₂₂, F.u, F.v, F.w)
@inline viscous_flux_wy(i, j, k, grid, clo::AID, K, clk, F, b) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, F, Σ₃₂, F.u, F.v, F.w)

@inline viscous_flux_ux(i, j, k, grid, clo::AHD, K, clk, F, b) = - ν_δᶜᶜᶜ(i, j, k, grid, clo, K, clk, F, F.u, F.v)
@inline viscous_flux_vx(i, j, k, grid, clo::AHD, K, clk, F, b) = - ν_ζᶠᶠᶜ(i, j, k, grid, clo, K, clk, F, F.u, F.v)
@inline viscous_flux_uy(i, j, k, grid, clo::AHD, K, clk, F, b) = + ν_ζᶠᶠᶜ(i, j, k, grid, clo, K, clk, F, F.u, F.v)   
@inline viscous_flux_vy(i, j, k, grid, clo::AHD, K, clk, F, b) = - ν_δᶜᶜᶜ(i, j, k, grid, clo, K, clk, F, F.u, F.v)
@inline viscous_flux_wx(i, j, k, grid, clo::AHD, K, clk, F, b) = - ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, F, ∂xᶠᶜᶠ, F.w)
@inline viscous_flux_wy(i, j, k, grid, clo::AHD, K, clk, F, b) = - ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, F, ∂yᶜᶠᶠ, F.w)

@inline viscous_flux_uz(i, j, k, grid, clo::AID, K, clk, F, b) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, F, Σ₁₃, F.u, F.v, F.w)
@inline viscous_flux_vz(i, j, k, grid, clo::AID, K, clk, F, b) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, F, Σ₂₃, F.u, F.v, F.w)
@inline viscous_flux_wz(i, j, k, grid, clo::AID, K, clk, F, b) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, F, Σ₃₃, F.u, F.v, F.w)

@inline viscous_flux_uz(i, j, k, grid, clo::AVD, K, clk, F, b) = - ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, F, ∂zᶠᶜᶠ, F.u)
@inline viscous_flux_vz(i, j, k, grid, clo::AVD, K, clk, F, b) = - ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, F, ∂zᶜᶠᶠ, F.v)
@inline viscous_flux_wz(i, j, k, grid, clo::AVD, K, clk, F, b) = - ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, F, ∂zᶜᶜᶜ, F.w)

# "Divergence damping"
@inline viscous_flux_ux(i, j, k, grid, clo::ADD, K, clk, F, b) = - ν_δᶜᶜᶜ(i, j, k, grid, clo, K, clk, F, F.u, F.v)
@inline viscous_flux_vy(i, j, k, grid, clo::ADD, K, clk, F, b) = - ν_δᶜᶜᶜ(i, j, k, grid, clo, K, clk, F, F.u, F.v)

#####
##### Diffusive fluxes
#####

const AIDorAHD = Union{AID, AHD}
const AIDorAVD = Union{AID, AVD}

@inline diffusive_flux_x(i, j, k, grid, cl::AIDorAHD, K, ::Val{id}, c, clk, F, b) where id = - κᶠᶜᶜ(i, j, k, grid, cl, K, Val(id), clk, F) * ∂xᶠᶜᶜ(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, cl::AIDorAHD, K, ::Val{id}, c, clk, F, b) where id = - κᶜᶠᶜ(i, j, k, grid, cl, K, Val(id), clk, F) * ∂yᶜᶠᶜ(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, cl::AIDorAVD, K, ::Val{id}, c, clk, F, b) where id = - κᶜᶜᶠ(i, j, k, grid, cl, K, Val(id), clk, F) * ∂zᶜᶜᶠ(i, j, k, grid, c)

#####
##### Support for VerticallyImplicit
#####

const VITD = VerticallyImplicitTimeDiscretization

@inline ivd_viscous_flux_uz(i, j, k, grid, closure, K, clock, F, b) = - ν_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, F, ∂xᶠᶜᶠ, F.w)
@inline ivd_viscous_flux_vz(i, j, k, grid, closure, K, clock, F, b) = - ν_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, F, ∂yᶜᶠᶠ, F.w)

# General functions (eg for vertically periodic)
@inline viscous_flux_uz(i, j, k, grid,  ::VITD, closure::AIDorAVD, args...) = ivd_viscous_flux_uz(i, j, k, grid, clo, args...)
@inline viscous_flux_vz(i, j, k, grid,  ::VITD, closure::AIDorAVD, args...) = ivd_viscous_flux_vz(i, j, k, grid, clo, args...)
@inline viscous_flux_wz(i, j, k, grid,  ::VITD, closure::AIDorAVD, args...) = zero(eltype(grid))
@inline diffusive_flux_z(i, j, k, grid, ::VITD, closure::AIDorAVD, args...) = zero(eltype(grid))
                  
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
                  zero(eltype(grid)))
end

@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::AIDorAVD, args...)
    return ifelse((k == 1) | (k == grid.Nz+1), 
                  diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  zero(eltype(grid)))
end

