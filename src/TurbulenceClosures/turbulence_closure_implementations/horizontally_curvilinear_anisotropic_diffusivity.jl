using Oceananigans.Operators: Δzᵃᵃᶜ

"""
    HorizontallyCurvilinearAnisotropicDiffusivity{N, K}

Holds viscosity and diffusivities for models with prescribed isotropic diffusivities.
"""
struct HorizontallyCurvilinearAnisotropicDiffusivity{NH, NZ, KH, KZ} <: AbstractTurbulenceClosure
    νh :: NH
    νz :: NZ
    κh :: KH
    κz :: KZ
end

const HCAD = HorizontallyCurvilinearAnisotropicDiffusivity

"""
    HorizontallyCurvilinearAnisotropicDiffusivity(; ν=0, κ=0)

Returns parameters for an horizontal diffusivity model on
curvilinear grids with viscosity `ν` and diffusivities `κ` for each tracer
field in `tracers`. `ν` and the fields of `κ` may be constants or functions
of `(x, y, z, t)`, and may represent molecular diffusivities in cases that all flow
features are explicitly resovled, or turbulent eddy diffusivities that model the effect of
unresolved, subgrid-scale turbulence.
`κ` may be a `NamedTuple` with fields corresponding
to each tracer, or a single number to be a applied to all tracers.
"""
function HorizontallyCurvilinearAnisotropicDiffusivity(FT=Float64; νh=0, κh=0, νz=0, κz=0)
    νh = convert_diffusivity(FT, νh)
    νz = convert_diffusivity(FT, νz)
    κh = convert_diffusivity(FT, κh)
    κz = convert_diffusivity(FT, κz)
    return HorizontallyCurvilinearAnisotropicDiffusivity(νh, νz, κh, κz)
end

function with_tracers(tracers, closure::HorizontallyCurvilinearAnisotropicDiffusivity)
    κh = tracer_diffusivities(tracers, closure.κh)
    κz = tracer_diffusivities(tracers, closure.κz)
    return HorizontallyCurvilinearAnisotropicDiffusivity(closure.νh, closure.νz, κh, κz)
end

calculate_diffusivities!(K, arch, grid, closure::HorizontallyCurvilinearAnisotropicDiffusivity, args...) = nothing

viscous_flux_ux(i, j, k, grid, clock, closure::HCAD, U, args...) = + ν_δᶜᶜᶜ(i, j, k, grid, clock, closure.νh, U.u, U.v)   
viscous_flux_uy(i, j, k, grid, clock, closure::HCAD, U, args...) = - ν_ζᶠᶠᶜ(i, j, k, grid, clock, closure.νh, U.u, U.v)   
viscous_flux_uz(i, j, k, grid, clock, closure::HCAD, U, args...) = + ν_uzᶠᶜᶠ(i, j, k, grid, clock, closure.νz, U.u)

viscous_flux_vx(i, j, k, grid, clock, closure::HCAD, U, args...) = ν_ζᶠᶠᶜ(i, j, k, grid, clock, closure.νh, U.u, U.v)
viscous_flux_vy(i, j, k, grid, clock, closure::HCAD, U, args...) = ν_δᶜᶜᶜ(i, j, k, grid, clock, closure.νh, U.u, U.v)
viscous_flux_vz(i, j, k, grid, clock, closure::HCAD, U, args...) = ν_vzᶜᶠᶠ(i, j, k, grid, clock, closure.νz, U.v)

@inline function diffusive_flux_x(i, j, k, grid, clock, closure::HCAD, c, ::Val{tracer_index}, args...) where tracer_index
    @inbounds κh = closure.κh[tracer_index]
    return diffusive_flux_x(i, j, k, grid, clock, κh, c)
end

@inline function diffusive_flux_y(i, j, k, grid, clock, closure::HCAD, c, ::Val{tracer_index}, args...) where tracer_index
    @inbounds κh = closure.κh[tracer_index]
    return diffusive_flux_y(i, j, k, grid, clock, κh, c)
end

@inline function diffusive_flux_z(i, j, k, grid, clock, closure::HCAD, c, ::Val{tracer_index}, args...) where tracer_index
    @inbounds κz = closure.κz[tracer_index]
    return diffusive_flux_z(i, j, k, grid, clock, κz, c)
end

Base.show(io::IO, closure::HorizontallyCurvilinearAnisotropicDiffusivity) =
    print(io, "AnisotropicDiffusivity: " *
              "(νh=$(closure.νh), νz=$(closure.νz)), " *
              "(κh=$(closure.κh), κz=$(closure.κz))")
