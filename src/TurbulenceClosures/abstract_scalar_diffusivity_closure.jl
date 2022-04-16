"""
    abstract type AbstractScalarDiffusivity <: AbstractTurbulenceClosure end

Abstract type for closures with *isotropic* diffusivities.
"""
abstract type AbstractScalarDiffusivity{TD, F} <: AbstractTurbulenceClosure{TD} end

"""
    struct ThreeDimensional end

Specifies a three-dimensionally-isotropic `ScalarDiffusivity`.
"""
struct ThreeDimensionalFormulation end

"""
    struct Horizontal end

Specifies a horizontally-isotropic, `VectorInvariant`, `ScalarDiffusivity`.
"""
struct HorizontalFormulation end

"""
    struct Vertical end

Specifies a `ScalarDiffusivity` acting only in the vertical direction.
"""
struct VerticalFormulation end

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

@inline νᶜᶜᶜ(i, j, k, grid, clo::ASD, K, clk)     = νᶜᶜᶜ(i, j, k, grid, clk, viscosity_location(clo), viscosity(clo, K)) 
@inline νᶠᶠᶜ(i, j, k, grid, clo::ASD, K, clk)     = νᶠᶠᶜ(i, j, k, grid, clk, viscosity_location(clo), viscosity(clo, K))
@inline νᶠᶜᶠ(i, j, k, grid, clo::ASD, K, clk)     = νᶠᶜᶠ(i, j, k, grid, clk, viscosity_location(clo), viscosity(clo, K))
@inline νᶜᶠᶠ(i, j, k, grid, clo::ASD, K, clk)     = νᶜᶠᶠ(i, j, k, grid, clk, viscosity_location(clo), viscosity(clo, K))

@inline κᶠᶜᶜ(i, j, k, grid, clo::ASD, K, id, clk) = κᶠᶜᶜ(i, j, k, grid, clk, diffusivity_location(clo), diffusivity(clo, K, id))
@inline κᶜᶠᶜ(i, j, k, grid, clo::ASD, K, id, clk) = κᶜᶠᶜ(i, j, k, grid, clk, diffusivity_location(clo), diffusivity(clo, K, id))
@inline κᶜᶜᶠ(i, j, k, grid, clo::ASD, K, id, clk) = κᶜᶜᶠ(i, j, k, grid, clk, diffusivity_location(clo), diffusivity(clo, K, id))

# Vertical and horizontal diffusivity
@inline νzᶜᶜᶜ(i, j, k, grid, clo::ASD, K, clk)     = νᶜᶜᶜ(i, j, k, grid, clo, K, clk) 
@inline νzᶠᶠᶜ(i, j, k, grid, clo::ASD, K, clock)     = νᶠᶠᶜ(i, j, k, grid, clo, K, clock) 
@inline νzᶠᶜᶠ(i, j, k, grid, clo::ASD, K, clock)     = νᶠᶜᶠ(i, j, k, grid, clo, K, clock) 
@inline νzᶜᶠᶠ(i, j, k, grid, clo::ASD, K, clock)     = νᶜᶠᶠ(i, j, k, grid, clo, K, clock) 
@inline κzᶠᶜᶜ(i, j, k, grid, clo::ASD, K, id, clock) = κᶠᶜᶜ(i, j, k, grid, clo, K, id, clock)
@inline κzᶜᶠᶜ(i, j, k, grid, clo::ASD, K, id, clock) = κᶜᶠᶜ(i, j, k, grid, clo, K, id, clock)
@inline κzᶜᶜᶠ(i, j, k, grid, clo::ASD, K, id, clock) = κᶜᶜᶠ(i, j, k, grid, clo, K, id, clock)

@inline νhᶜᶜᶜ(i, j, k, grid, clo::ASD, K, clock)     = νᶜᶜᶜ(i, j, k, grid, clo, K, clock) 
@inline νhᶠᶠᶜ(i, j, k, grid, clo::ASD, K, clock)     = νᶠᶠᶜ(i, j, k, grid, clo, K, clock) 
@inline νhᶠᶜᶠ(i, j, k, grid, clo::ASD, K, clock)     = νᶠᶜᶠ(i, j, k, grid, clo, K, clock) 
@inline νhᶜᶠᶠ(i, j, k, grid, clo::ASD, K, clock)     = νᶜᶠᶠ(i, j, k, grid, clo, K, clock) 
@inline κhᶠᶜᶜ(i, j, k, grid, clo::ASD, K, id, clock) = κᶠᶜᶜ(i, j, k, grid, clo, K, id, clock)
@inline κhᶜᶠᶜ(i, j, k, grid, clo::ASD, K, id, clock) = κᶜᶠᶜ(i, j, k, grid, clo, K, id, clock)
@inline κhᶜᶜᶠ(i, j, k, grid, clo::ASD, K, id, clock) = κᶜᶜᶠ(i, j, k, grid, clo, K, id, clock)

for (dir, Clo) in zip((:h, :z), (:AVD, :AHD))
    for code in (:ᶜᶜᶜ, :ᶠᶠᶜ, :ᶠᶜᶠ, :ᶜᶠᶠ)
        ν = Symbol(:ν, dir, code)
        @eval begin
            @inline $ν(i, j, k, grid, clo::$Clo, args...) = zero(eltype(grid))
        end
    end

    for code in (:ᶠᶜᶜ, :ᶜᶠᶜ, :ᶜᶜᶠ)
        κ = Symbol(:κ, dir, code)
        @eval begin
            @inline $κ(i, j, k, grid, clo::$Clo, args...) = zero(eltype(grid))
        end
    end
end

const F = Face
const C = Center

@inline z_diffusivity(i, j, k, grid, ::F, ::C, ::C, clo::ASD, K, id, clock) = κzᶠᶜᶜ(i, j, k, grid, clo, K, id, clock)
@inline z_diffusivity(i, j, k, grid, ::C, ::F, ::C, clo::ASD, K, id, clock) = κzᶜᶠᶜ(i, j, k, grid, clo, K, id, clock)
@inline z_diffusivity(i, j, k, grid, ::C, ::C, ::F, clo::ASD, K, id, clock) = κzᶜᶜᶠ(i, j, k, grid, clo, K, id, clock)

@inline h_diffusivity(i, j, k, grid, ::F, ::C, ::C, clo::ASD, K, id, clock) = κhᶠᶜᶜ(i, j, k, grid, clo, K, id, clock)
@inline h_diffusivity(i, j, k, grid, ::C, ::F, ::C, clo::ASD, K, id, clock) = κhᶜᶠᶜ(i, j, k, grid, clo, K, id, clock)
@inline h_diffusivity(i, j, k, grid, ::C, ::C, ::F, clo::ASD, K, id, clock) = κhᶜᶜᶠ(i, j, k, grid, clo, K, id, clock)

# "diffusivity" with "nothing" index => viscosity of course
@inline z_diffusivity(i, j, k, grid, ::C, ::C, ::C, clo::ASD, K, ::Nothing, clock) = νzᶜᶜᶜ(i, j, k, grid, clo, K, clock)
@inline z_diffusivity(i, j, k, grid, ::F, ::F, ::C, clo::ASD, K, ::Nothing, clock) = νzᶠᶠᶜ(i, j, k, grid, clo, K, clock)
@inline z_diffusivity(i, j, k, grid, ::F, ::C, ::F, clo::ASD, K, ::Nothing, clock) = νzᶠᶜᶠ(i, j, k, grid, clo, K, clock)
@inline z_diffusivity(i, j, k, grid, ::C, ::F, ::F, clo::ASD, K, ::Nothing, clock) = νzᶜᶠᶠ(i, j, k, grid, clo, K, clock)

@inline h_diffusivity(i, j, k, grid, ::C, ::C, ::C, clo::ASD, K, ::Nothing, clock) = νhᶜᶜᶜ(i, j, k, grid, clo, K, clock)
@inline h_diffusivity(i, j, k, grid, ::F, ::F, ::C, clo::ASD, K, ::Nothing, clock) = νhᶠᶠᶜ(i, j, k, grid, clo, K, clock)
@inline h_diffusivity(i, j, k, grid, ::F, ::C, ::F, clo::ASD, K, ::Nothing, clock) = νhᶠᶜᶠ(i, j, k, grid, clo, K, clock)
@inline h_diffusivity(i, j, k, grid, ::C, ::F, ::F, clo::ASD, K, ::Nothing, clock) = νhᶜᶠᶠ(i, j, k, grid, clo, K, clock)

#####
##### Stress divergences
#####

@inline viscous_flux_ux(i, j, k, grid, clo::AID, K, U, C, clk, b) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, Σ₁₁, U.u, U.v, U.w)
@inline viscous_flux_vx(i, j, k, grid, clo::AID, K, U, C, clk, b) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, clo, K, clk, Σ₂₁, U.u, U.v, U.w)
@inline viscous_flux_wx(i, j, k, grid, clo::AID, K, U, C, clk, b) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, Σ₃₁, U.u, U.v, U.w)
@inline viscous_flux_uy(i, j, k, grid, clo::AID, K, U, C, clk, b) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, clo, K, clk, Σ₁₂, U.u, U.v, U.w)
@inline viscous_flux_vy(i, j, k, grid, clo::AID, K, U, C, clk, b) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, Σ₂₂, U.u, U.v, U.w)
@inline viscous_flux_wy(i, j, k, grid, clo::AID, K, U, C, clk, b) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, Σ₃₂, U.u, U.v, U.w)

@inline viscous_flux_ux(i, j, k, grid, clo::AHD, K, U, C, clk, b) = - ν_δᶜᶜᶜ(i, j, k, grid, clo, K, clk, U.u, U.v)
@inline viscous_flux_vx(i, j, k, grid, clo::AHD, K, U, C, clk, b) = - ν_ζᶠᶠᶜ(i, j, k, grid, clo, K, clk, U.u, U.v)
@inline viscous_flux_uy(i, j, k, grid, clo::AHD, K, U, C, clk, b) = + ν_ζᶠᶠᶜ(i, j, k, grid, clo, K, clk, U.u, U.v)   
@inline viscous_flux_vy(i, j, k, grid, clo::AHD, K, U, C, clk, b) = - ν_δᶜᶜᶜ(i, j, k, grid, clo, K, clk, U.u, U.v)
@inline viscous_flux_wx(i, j, k, grid, clo::AHD, K, U, C, clk, b) = - ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, ∂xᶠᶜᶠ, U.w)
@inline viscous_flux_wy(i, j, k, grid, clo::AHD, K, U, C, clk, b) = - ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, ∂yᶜᶠᶠ, U.w)

@inline viscous_flux_uz(i, j, k, grid, clo::AID, K, U, C, clk, b) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, Σ₁₃, U.u, U.v, U.w)
@inline viscous_flux_vz(i, j, k, grid, clo::AID, K, U, C, clk, b) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, Σ₂₃, U.u, U.v, U.w)
@inline viscous_flux_wz(i, j, k, grid, clo::AID, K, U, C, clk, b) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, Σ₃₃, U.u, U.v, U.w)

@inline viscous_flux_uz(i, j, k, grid, clo::AVD, K, U, C, clk, b) = - ν_σᶠᶜᶠ(i, j, k, grid, clo, K, clk, ∂zᶠᶜᶠ, U.u)
@inline viscous_flux_vz(i, j, k, grid, clo::AVD, K, U, C, clk, b) = - ν_σᶜᶠᶠ(i, j, k, grid, clo, K, clk, ∂zᶜᶠᶠ, U.v)
@inline viscous_flux_wz(i, j, k, grid, clo::AVD, K, U, C, clk, b) = - ν_σᶜᶜᶜ(i, j, k, grid, clo, K, clk, ∂zᶜᶜᶜ, U.w)

#####
##### Diffusive fluxes
#####

const AIDorAHD = Union{AID, AHD}
const AIDorAVD = Union{AID, AVD}

@inline diffusive_flux_x(i, j, k, grid, cl::AIDorAHD, K, ::Val{id}, U, C, clk, b) where id = - κᶠᶜᶜ(i, j, k, grid, cl, K, Val(id), clk) * ∂xᶠᶜᶜ(i, j, k, grid, C[id])
@inline diffusive_flux_y(i, j, k, grid, cl::AIDorAHD, K, ::Val{id}, U, C, clk, b) where id = - κᶜᶠᶜ(i, j, k, grid, cl, K, Val(id), clk) * ∂yᶜᶠᶜ(i, j, k, grid, C[id])
@inline diffusive_flux_z(i, j, k, grid, cl::AIDorAVD, K, ::Val{id}, U, C, clk, b) where id = - κᶜᶜᶠ(i, j, k, grid, cl, K, Val(id), clk) * ∂zᶜᶜᶠ(i, j, k, grid, C[id])

#####
##### Zero out not used fluxes
#####

for (dir, Clo) in zip((:x, :y, :z), (:AVD, :AVD, :AHD))
    diffusive_flux = Symbol(:diffusive_flux_, dir)
    viscous_flux_u = Symbol(:viscous_flux_u, dir)
    viscous_flux_v = Symbol(:viscous_flux_v, dir)
    viscous_flux_w = Symbol(:viscous_flux_w, dir)
    @eval begin
        @inline $diffusive_flux(i, j, k, grid, closure::$Clo, K, ::Val, args...) = zero(eltype(grid))
        @inline $viscous_flux_u(i, j, k, grid, closure::$Clo, args...) = zero(eltype(grid))
        @inline $viscous_flux_v(i, j, k, grid, closure::$Clo, args...) = zero(eltype(grid))
        @inline $viscous_flux_w(i, j, k, grid, closure::$Clo, args...) = zero(eltype(grid))
    end
end

#####
##### Support for VerticallyImplicit
#####

const VITD = VerticallyImplicitTimeDiscretization

@inline ivd_viscous_flux_uz(i, j, k, grid, closure, K, U, C, clock, b) = - ν_σᶠᶜᶠ(i, j, k, grid, closure, K, clock, ∂xᶠᶜᶠ, U.w)
@inline ivd_viscous_flux_vz(i, j, k, grid, closure, K, U, C, clock, b) = - ν_σᶜᶠᶠ(i, j, k, grid, closure, K, clock, ∂yᶜᶠᶠ, U.w)

# General functions (eg for vertically periodic)
@inline viscous_flux_uz(i, j, k, grid,  ::VITD, clo::AIDorAVD, args...) = ivd_viscous_flux_uz(i, j, k, grid, clo, args...)
@inline viscous_flux_vz(i, j, k, grid,  ::VITD, clo::AIDorAVD, args...) = ivd_viscous_flux_vz(i, j, k, grid, clo, args...)
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

