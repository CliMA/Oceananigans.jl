import Oceananigans.Grids: required_halo_size

"""
    AnisotropicBiharmonicDiffusivity{FT, KH, KZ}

Parameters for anisotropic biharmonic diffusivity models.
"""
struct AnisotropicBiharmonicDiffusivity{FT, KX, KY, KZ} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization}
    νx :: FT
    νy :: FT
    νz :: FT
    κx :: KX
    κy :: KY
    κz :: KZ
end

"""
    AnisotropicBiharmonicDiffusivity(FT=Float64; νx=0, νy=0, νz=0, νh=nothing, κx=0, κy=0, κz=0, κh=nothing)

Returns parameters for a fourth-order, anisotropic biharmonic diffusivity closure with
constant x-, y, and z-direction biharmonic viscosities `νx`, `νy`, and `νz`,
and constant x-, y, and z-direction biharmonic diffusivities `κx`, `κy`, and `κz`,
`κx`, `κy`, and `κz` may be `NamedTuple`s with fields corresponding
to each tracer, or a single number to be a applied to all tracers.

If `νh` or `κh` are provided, then `νx = νy = νh` or `κx = κy = κh`.

The tracer flux divergence associated with an anisotropic biharmonic diffusivity is, for example

```math
    ∂ᵢ κᵢⱼ ∂ⱼc = - [κx ∂⁴x + κy ∂⁴y + κz ∂⁴z] c
```

"""
function AnisotropicBiharmonicDiffusivity(FT=Float64; νx=0, νy=0, νz=0, κx=0, κy=0, κz=0, νh=nothing, κh=nothing)
    if !isnothing(νh)
        νx = νh
        νy = νh
    end

    if !isnothing(κh)
        κx = κh
        κy = κh
    end
    
    κx = convert_diffusivity(FT, κx)
    κy = convert_diffusivity(FT, κy)
    κz = convert_diffusivity(FT, κz)

    return AnisotropicBiharmonicDiffusivity(FT(νx), FT(νy), FT(νz), κx, κy, κz)
end

required_halo_size(closure::AnisotropicBiharmonicDiffusivity) = 2
                                            
function with_tracers(tracers, closure::AnisotropicBiharmonicDiffusivity)
    κx = tracer_diffusivities(tracers, closure.κx)
    κy = tracer_diffusivities(tracers, closure.κy)
    κz = tracer_diffusivities(tracers, closure.κz)
    return AnisotropicBiharmonicDiffusivity(closure.νx, closure.νy, closure.νz, κx, κy, κz)
end

calculate_diffusivities!(K, arch, grid, closure::AnisotropicBiharmonicDiffusivity, args...) = nothing

@inline ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure::AnisotropicBiharmonicDiffusivity, clock, U, args...) = (
    - closure.νx * ∂⁴xᶠᵃᵃ(i, j, k, grid, U.u)
    - closure.νy * ∂⁴yᵃᶜᵃ(i, j, k, grid, U.u)
    - closure.νz * ∂⁴zᵃᵃᶜ(i, j, k, grid, U.u)
    )

@inline ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure::AnisotropicBiharmonicDiffusivity, clock, U, args...) = (
    - closure.νx * ∂⁴xᶜᵃᵃ(i, j, k, grid, U.v)
    - closure.νy * ∂⁴yᵃᶠᵃ(i, j, k, grid, U.v)
    - closure.νz * ∂⁴zᵃᵃᶜ(i, j, k, grid, U.v)
    )

@inline ∂ⱼ_τ₃ⱼ(i, j, k, grid, closure::AnisotropicBiharmonicDiffusivity, clock, U, args...) = (
    - closure.νx * ∂⁴xᶜᵃᵃ(i, j, k, grid, U.w)
    - closure.νy * ∂⁴yᵃᶜᵃ(i, j, k, grid, U.w)
    - closure.νz * ∂⁴zᵃᵃᶠ(i, j, k, grid, U.w)
    )

@inline function ∇_dot_qᶜ(i, j, k, grid, closure::AnisotropicBiharmonicDiffusivity,
                          c, ::Val{tracer_index}, args...) where tracer_index

    @inbounds κx = closure.κx[tracer_index]
    @inbounds κy = closure.κy[tracer_index]
    @inbounds κz = closure.κz[tracer_index]

    return (- κx * ∂⁴xᶜᵃᵃ(i, j, k, grid, c)
            - κy * ∂⁴yᵃᶜᵃ(i, j, k, grid, c)
            - κz * ∂⁴zᵃᵃᶜ(i, j, k, grid, c)
           )
end
