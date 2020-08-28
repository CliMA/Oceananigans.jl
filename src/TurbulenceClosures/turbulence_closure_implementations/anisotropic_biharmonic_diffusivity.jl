"""
    AnisotropicBiharmonicDiffusivity{FT, KH, KZ}

Parameters for anisotropic biharmonic diffusivity models.
"""
struct AnisotropicBiharmonicDiffusivity{FT, KH, KZ} <: AbstractTensorDiffusivity
    νh :: FT
    νz :: FT
    κh :: KH
    κz :: KZ
    function AnisotropicBiharmonicDiffusivity{FT}(νh, νz, κh, κz) where FT
        return new{FT, typeof(κh), typeof(κz)}(νh, νz, convert_diffusivity(FT, κh), convert_diffusivity(FT, κz))
    end
end

"""
    AnisotropicBiharmonicDiffusivity(FT=Float64; νh, νz, κh, κz)

Returns parameters for a fourth-order, anisotropic biharmonic diffusivity closure with
constant horizontal and vertical biharmonic viscosities `νh`, `νz` and constant horizontal
and vertical tracer biharmonic diffusivities `κh`, `κz`.
`κh` and `κz` may be `NamedTuple`s with fields corresponding
to each tracer, or a single number to be a applied to all tracers.
The tracer flux divergence associated with an anisotropic biharmonic diffusivity is, for example

```math
    ∂ᵢ κᵢⱼ ∂ⱼc = (κh ∇⁴h + κz ∂⁴z) c
```

"""
AnisotropicBiharmonicDiffusivity(FT=Float64; νh=0, νz=0, κh=0, κz=0) =
    AnisotropicBiharmonicDiffusivity{FT}(νh, νz, κh, κz)

function with_tracers(tracers, closure::AnisotropicBiharmonicDiffusivity{FT}) where FT
    κh = tracer_diffusivities(tracers, closure.κh)
    κz = tracer_diffusivities(tracers, closure.κz)
    return AnisotropicBiharmonicDiffusivity{FT}(closure.νh, closure.νz, κh, κz)
end

calculate_diffusivities!(K, arch, grid, closure::AnisotropicBiharmonicDiffusivity, args...) = nothing

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, clock, closure::AnisotropicBiharmonicDiffusivity, U, args...) = (
    - closure.νh * ∇⁴hᶠᶜᵃ(i, j, k, grid, U.u)
    - closure.νz * ∂⁴zᵃᵃᶜ(i, j, k, grid, U.u)
    )

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, clock, closure::AnisotropicBiharmonicDiffusivity, U, args...) = (
    - closure.νh * ∇⁴hᶜᶠᵃ(i, j, k, grid, U.v)
    - closure.νz * ∂⁴zᵃᵃᶜ(i, j, k, grid, U.v)
    )

@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, clock, closure::AnisotropicBiharmonicDiffusivity, U, args...) = (
    - closure.νh * ∇⁴hᶜᶜᵃ(i, j, k, grid, U.w)
    - closure.νz * ∂⁴zᵃᵃᶠ(i, j, k, grid, U.w)
    )

@inline function ∇_κ_∇c(i, j, k, grid, clock, closure::AnisotropicBiharmonicDiffusivity,
                        c, ::Val{tracer_index}, args...) where tracer_index

    @inbounds κh = closure.κh[tracer_index]
    @inbounds κz = closure.κz[tracer_index]

    return (- κh * ∇⁴hᶜᶜᵃ(i, j, k, grid, c)
            - κz * ∂⁴zᵃᵃᶜ(i, j, k, grid, c)
           )
end
