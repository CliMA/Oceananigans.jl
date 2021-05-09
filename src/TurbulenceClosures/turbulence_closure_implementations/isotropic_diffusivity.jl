import Oceananigans.Grids: required_halo_size

"""
    IsotropicDiffusivity{N, K}

Holds viscosity and diffusivities for models with prescribed isotropic diffusivities.
"""
struct IsotropicDiffusivity{N, K} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization}
    ν :: N
    κ :: K
end

"""
    IsotropicDiffusivity(; ν=ν₀, κ=κ₀)

Returns parameters for an isotropic diffusivity model with viscosity `ν`
and thermal diffusivities `κ` for each tracer field in `tracers`
`ν` and the fields of `κ` may be constants or functions of `(x, y, z, t)`, and
may represent molecular diffusivities in cases that all flow
features are explicitly resovled, or turbulent eddy diffusivities that model the effect of
unresolved, subgrid-scale turbulence.
`κ` may be a `NamedTuple` with fields corresponding
to each tracer, or a single number to be a applied to all tracers.

By default, a molecular viscosity of `ν₀ = 1.05×10⁻⁶` m² s⁻¹ and a molecular thermal
diffusivity of `κ₀ = 1.46×10⁻⁷` m² s⁻¹ is used for each tracer. These molecular values are
the approximate viscosity and thermal diffusivity for seawater at 20°C and 35 psu,
according to Sharqawy et al., "Thermophysical properties of seawater: A review of existing
correlations and data" (2010).
"""
function IsotropicDiffusivity(FT=Float64; ν=ν₀, κ=κ₀)
    if ν isa Number && κ isa Number
        κ = convert_diffusivity(FT, κ)
        return IsotropicDiffusivity(FT(ν), κ)
    else
        return IsotropicDiffusivity(ν, κ)
    end
end

required_halo_size(closure::IsotropicDiffusivity) = 1 
 
function with_tracers(tracers, closure::IsotropicDiffusivity)
    κ = tracer_diffusivities(tracers, closure.κ)
    return IsotropicDiffusivity(closure.ν, κ)
end

calculate_diffusivities!(K, arch, grid, closure::IsotropicDiffusivity, args...) = nothing

@inline function diffusive_flux_x(i, j, k, grid, clock, closure::IsotropicDiffusivity, c, ::Val{tracer_index}, args...) where tracer_index
    @inbounds κ = closure.κ[tracer_index]
    return diffusive_flux_x(i, j, k, grid, clock, κ, c)
end

@inline function diffusive_flux_y(i, j, k, grid, clock, closure::IsotropicDiffusivity, c, ::Val{tracer_index}, args...) where tracer_index
    @inbounds κ = closure.κ[tracer_index]
    return diffusive_flux_y(i, j, k, grid, clock, κ, c)
end

@inline function diffusive_flux_z(i, j, k, grid, clock, closure::IsotropicDiffusivity, c, ::Val{tracer_index}, args...) where tracer_index
    @inbounds κ = closure.κ[tracer_index]
    return diffusive_flux_z(i, j, k, grid, clock, κ, c)
end

const ID = IsotropicDiffusivity

@inline viscous_flux_ux(i, j, k, grid, clock, closure::ID, U, args...) = @inbounds viscous_flux_ux(i, j, k, grid, clock, closure.ν, U[1])
@inline viscous_flux_uy(i, j, k, grid, clock, closure::ID, U, args...) = @inbounds viscous_flux_uy(i, j, k, grid, clock, closure.ν, U[1])  
@inline viscous_flux_uz(i, j, k, grid, clock, closure::ID, U, args...) = @inbounds viscous_flux_uz(i, j, k, grid, clock, closure.ν, U[1])

@inline viscous_flux_vx(i, j, k, grid, clock, closure::ID, U, args...) = @inbounds viscous_flux_vx(i, j, k, grid, clock, closure.ν, U[2])
@inline viscous_flux_vy(i, j, k, grid, clock, closure::ID, U, args...) = @inbounds viscous_flux_vy(i, j, k, grid, clock, closure.ν, U[2])  
@inline viscous_flux_vz(i, j, k, grid, clock, closure::ID, U, args...) = @inbounds viscous_flux_vz(i, j, k, grid, clock, closure.ν, U[2])

@inline viscous_flux_wx(i, j, k, grid, clock, closure::ID, U, args...) = @inbounds viscous_flux_wx(i, j, k, grid, clock, closure.ν, U[3])
@inline viscous_flux_wy(i, j, k, grid, clock, closure::ID, U, args...) = @inbounds viscous_flux_wy(i, j, k, grid, clock, closure.ν, U[3])  
@inline viscous_flux_wz(i, j, k, grid, clock, closure::ID, U, args...) = @inbounds viscous_flux_wz(i, j, k, grid, clock, closure.ν, U[3])
                        
Base.show(io::IO, closure::IsotropicDiffusivity) =
    print(io, "IsotropicDiffusivity: ν=$(closure.ν), κ=$(closure.κ)")
