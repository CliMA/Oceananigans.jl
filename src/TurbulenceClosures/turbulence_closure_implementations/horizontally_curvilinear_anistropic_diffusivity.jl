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

@inline function ∇_κ_∇c(i, j, k, grid, clock, closure::HorizontallyCurvilinearAnisotropicDiffusivity,
                        c, ::Val{tracer_index}, args...) where tracer_index

    @inbounds κh = closure.κh[tracer_index]
    @inbounds κz = closure.κz[tracer_index]

    return ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, clock, κh, κh, κz, c)
end

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, clock, closure::HorizontallyCurvilinearAnisotropicDiffusivity, U, args...) = (
      δxᶠᵃᵃ(i, j, k, grid, ν_δᶜᶜᶜ, clock, closure.νh, U.u, U.v) / Δxᶠᶜᵃ(i, j, k, grid)
    - δyᵃᶜᵃ(i, j, k, grid, ν_ζᶠᶠᶜ, clock, closure.νh, U.u, U.v) / Δyᶠᶜᵃ(i, j, k, grid)
    + δzᵃᵃᶜ(i, j, k, grid, ν_uzᶠᶜᶠ, clock, closure.νz, U.u)     / Δzᵃᵃᶜ(i, j, k, grid)
)

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, clock, closure::HorizontallyCurvilinearAnisotropicDiffusivity, U, args...) = (
      δyᵃᶠᵃ(i, j, k, grid, ν_δᶜᶜᶜ, clock, closure.νh, U.u, U.v) / Δyᶜᶠᵃ(i, j, k, grid)
    + δxᶜᵃᵃ(i, j, k, grid, ν_ζᶠᶠᶜ, clock, closure.νh, U.u, U.v) / Δxᶜᶠᵃ(i, j, k, grid)
    + δzᵃᵃᶜ(i, j, k, grid, ν_vzᶜᶠᶠ, clock, closure.νz, U.v)     / Δzᵃᵃᶜ(i, j, k, grid)
)

Base.show(io::IO, closure::HorizontallyCurvilinearAnisotropicDiffusivity) =
    print(io, "AnisotropicDiffusivity: " *
              "(νh=$(closure.νh), νz=$(closure.νz)), " *
              "(κh=$(closure.κh), κz=$(closure.κz))")
