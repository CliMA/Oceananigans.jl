import Oceananigans.Grids: required_halo_size

struct ScalarDiffusivity{TD, Dir, N, K} <: AbstractScalarDiffusivity{TD, Dir}
    ν :: N
    κ :: K

    function ScalarDiffusivity{TD, Dir}(ν::N, κ::K) where {TD, N, K}
        return new{TD, Dir, N, K}(ν, κ)
    end
end

struct ThreeDimensional end

struct Horizontal end

struct Vertical end

"""
    ScalarDiffusivity([FT=Float64;]
                         ν=0, κ=0, time_discretization = ExplicitTimeDiscretization())

Returns parameters for an isotropic diffusivity model with viscosity `ν`
and thermal diffusivities `κ` for each tracer field in `tracers`
`ν` and the fields of `κ` may be constants, arrays, fields, or
functions of `(x, y, z, t)`.

`κ` may be a `NamedTuple` with fields corresponding to each tracer, or a
single number to be a applied to all tracers.
"""
function ScalarDiffusivity(FT=Float64, direction::Dir=ThreeDimensional();
                              ν=0, κ=0, time_discretization::TD = ExplicitTimeDiscretization()) where {TD, Dir}

    if ν isa Number && κ isa Number
        κ = convert_diffusivity(FT, κ)
        return ScalarDiffusivity{TD, Dir}(FT(ν), κ)
    else
        return ScalarDiffusivity{TD, Dir}(ν, κ)
    end
end

required_halo_size(closure::ScalarDiffusivity) = 1 
 
function with_tracers(tracers, closure::ScalarDiffusivity{TD, Dir}) where {TD, Dir}
    κ = tracer_diffusivities(tracers, closure.κ)
    return ScalarDiffusivity{TD, Dir}(closure.ν, κ)
end

calculate_diffusivities!(diffusivities, closure::ScalarDiffusivity, args...) = nothing

@inline diffusivity(closure::ScalarDiffusivity, ::Val{tracer_index}, args...) where tracer_index = closure.κ[tracer_index]
@inline viscosity(closure::ScalarDiffusivity, args...) = closure.ν
                        
Base.show(io::IO, closure::ScalarDiffusivity{TD, Dir}) = 
    print(io, "ScalarDiffusivity:\n",
              "ν=$(closure.ν), κ=$(closure.κ)"
              "time discretization: $(time_discretization(closure))\n",
              "direction: $TD")

#####
##### Stress divergences
#####

const ID = ScalarDiffusivity{<:Any, <:ThreeDimensional}
const HD = ScalarDiffusivity{<:Any, <:Horizontal}
const VD = ScalarDiffusivity{<:Any, <:Vertical}

@inline viscous_flux_ux(i, j, k, grid, closure::ID, clock, U, args...) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, args...), Σ₁₁, U.u, U.v, U.w)
@inline viscous_flux_vx(i, j, k, grid, closure::ID, clock, U, args...) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, args...), Σ₂₁, U.u, U.v, U.w)
@inline viscous_flux_wx(i, j, k, grid, closure::ID, clock, U, args...) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, args...), Σ₃₁, U.u, U.v, U.w)

@inline viscous_flux_ux(i, j, k, grid, closure::HD, clock, U, args...) = - ν_δᶜᶜᶜ(i, j, k, grid, clock, closure.νh, U.u, U.v)   
@inline viscous_flux_vx(i, j, k, grid, closure::HD, clock, U, args...) = - ν_ζᶠᶠᶜ(i, j, k, grid, clock, closure.νh, U.u, U.v)

@inline viscous_flux_uy(i, j, k, grid, closure::ID, clock, U, args...) = - 2 * ν_σᶠᶠᶜ(i, j, k, grid, clock, viscosity(closure, args...), Σ₁₂, U.u, U.v, U.w)
@inline viscous_flux_vy(i, j, k, grid, closure::ID, clock, U, args...) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, args...), Σ₂₂, U.u, U.v, U.w)
@inline viscous_flux_wy(i, j, k, grid, closure::ID, clock, U, args...) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, args...), Σ₃₂, U.u, U.v, U.w)

@inline viscous_flux_uy(i, j, k, grid, closure::HD, clock, U, args...) = + ν_ζᶠᶠᶜ(i, j, k, grid, clock, closure.νh, U.u, U.v)   
@inline viscous_flux_vy(i, j, k, grid, closure::HD, clock, U, args...) = - ν_δᶜᶜᶜ(i, j, k, grid, clock, closure.νh, U.u, U.v)

@inline viscous_flux_uz(i, j, k, grid, closure::Union{ID, VD}, clock, U, args...) = - 2 * ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, args...), Σ₁₃, U.u, U.v, U.w)
@inline viscous_flux_vz(i, j, k, grid, closure::Union{ID, VD}, clock, U, args...) = - 2 * ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, args...), Σ₂₃, U.u, U.v, U.w)
@inline viscous_flux_wz(i, j, k, grid, closure::Union{ID, VD}, clock, U, args...) = - 2 * ν_σᶜᶜᶜ(i, j, k, grid, clock, viscosity(closure, args...), Σ₃₃, U.u, U.v, U.w)

#####
##### Diffusive fluxes
#####

@inline diffusive_flux_x(i, j, k, grid, closure::Union{ID, HD}, c, c_idx, clock, args...) = diffusive_flux_x(i, j, k, grid, clock, diffusivity(closure, c_idx, args...), c)
@inline diffusive_flux_y(i, j, k, grid, closure::Union{ID, HD}, c, c_idx, clock, args...) = diffusive_flux_y(i, j, k, grid, clock, diffusivity(closure, c_idx, args...), c)
@inline diffusive_flux_z(i, j, k, grid, closure::Union{ID, VD}, c, c_idx, clock, args...) = diffusive_flux_z(i, j, k, grid, clock, diffusivity(closure, c_idx, args...), c)

#####
##### Zero out not used fluxes
#####

for (dir, closure) in zip((:x, :y, :z), (:VD, :VD, :HD))
    diffusive_flux = Symbol(:diffusive_flux_, dir)
    viscous_flux_u = Symbol(:viscous_flux_u, dir)
    viscous_flux_v = Symbol(:viscous_flux_v, dir)
    @eval begin
        @inline $diffusive_flux(i, j, k, grid, closure::$closure, c, c_idx, clock, args...) = zero(eltype(grid))
        @inline $viscous_flux_u(i, j, k, grid, closure::$closure, c, c_idx, clock, args...) = zero(eltype(grid))
        @inline $viscous_flux_v(i, j, k, grid, closure::$closure, c, c_idx, clock, args...) = zero(eltype(grid))
    end
end

#####
##### Support for VerticallyImplicitTimeDiscretization
#####

const VITD = VerticallyImplicitTimeDiscretization
const VerticallyBoundedGrid{FT} = AbstractGrid{FT, <:Any, <:Any, <:Bounded}

  @inline z_viscosity(closure::Union{ID, VD}, args...)        = viscosity(closure, args...)
@inline z_diffusivity(closure::Union{ID, VD}, c_idx, args...) = diffusivity(closure, c_idx, args...)

@inline ivd_viscous_flux_uz(i, j, k, grid, closure, clock, U, args...) = - ν_σᶠᶜᶠ(i, j, k, grid, clock, viscosity(closure, args...), ∂xᶠᶜᶠ, U.w)
@inline ivd_viscous_flux_vz(i, j, k, grid, closure, clock, U, args...) = - ν_σᶜᶠᶠ(i, j, k, grid, clock, viscosity(closure, args...), ∂yᶜᶠᶠ, U.w)

# General functions (eg for vertically periodic)
@inline viscous_flux_uz(i, j, k, grid,  ::VITD, closure::Union{ID, VD}, args...) = ivd_viscous_flux_uz(i, j, k, grid, closure, args...)
@inline viscous_flux_vz(i, j, k, grid,  ::VITD, closure::Union{ID, VD}, args...) = ivd_viscous_flux_vz(i, j, k, grid, closure, args...)
@inline viscous_flux_wz(i, j, k, grid,  ::VITD, closure::Union{ID, VD}, args...) = zero(eltype(grid))
@inline diffusive_flux_z(i, j, k, grid, ::VITD, closure::Union{ID, VD}, clock, args...) = zero(eltype(grid))
                  
# Vertically bounded grids
#
# For vertically bounded grids, we calculate _explicit_ fluxes on the boundaries, 
# and elide the implicit vertical flux component everywhere else. This is consistent
# with the formulation of the tridiagonal solver, which requires explicit treatment
# of boundary contributions (eg boundary contributions must be moved to the right
# hand side of the tridiagonal system).

@inline function viscous_flux_uz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::Union{ID, VD}, args...)
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_uz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  ivd_viscous_flux_uz(i, j, k, grid, closure, args...))
end

@inline function viscous_flux_vz(i, j, k, grid::VerticallyBoundedGrid, ::VITD, closure::Union{ID, VD}, args...)
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_vz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  ivd_viscous_flux_vz(i, j, k, grid, closure, args...))
end

@inline function viscous_flux_wz(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::Union{ID, VD}, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  viscous_flux_wz(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  zero(FT))
end

@inline function diffusive_flux_z(i, j, k, grid::VerticallyBoundedGrid{FT}, ::VITD, closure::Union{ID, VD}, args...) where FT
    return ifelse(k == 1 || k == grid.Nz+1, 
                  diffusive_flux_z(i, j, k, grid, ExplicitTimeDiscretization(), closure, args...),
                  zero(FT))
end