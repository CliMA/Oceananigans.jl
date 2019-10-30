"""
    AnisotropicBiharmonicDiffusivity{FT, KH, KV}

Parameters for constant anisotropic diffusivity models.
"""
struct AnisotropicBiharmonicDiffusivity{FT, KH, KV} <: TensorDiffusivity{FT}
    νh :: FT
    νv :: FT
    κh :: KH
    κv :: KV
    function AnisotropicBiharmonicDiffusivity{FT}(νh, νv, κh, κv) where FT
        return new{FT, typeof(κh), typeof(κv)}(νh, νv, convert_diffusivity(FT, κh), convert_diffusivity(FT, κv))
    end
end

"""
    AnisotropicBiharmonicDiffusivity(; νh, νv, κh, κv)

Returns parameters for a anisotropic biharmonic diffusivity closure with constant horizontal
and vertical biharmonic viscosities `νh`, `νv` and constant horizontal and vertical thermal 
biharmonic diffusivities `κh`, `κv`. 
"""
AnisotropicBiharmonicDiffusivity(FT=Float64; νh, νv, κh, κv) =
    AnisotropicBiharmonicDiffusivity{FT}(νh, νv, κh, κv)

function with_tracers(tracers, closure::AnisotropicBiharmonicDiffusivity{FT}) where FT
    κh = tracer_diffusivities(tracers, closure.κh)
    κv = tracer_diffusivities(tracers, closure.κv)
    return AnisotropicBiharmonicDiffusivity{FT}(closure.νh, closure.νv, κh, κv)
end

calculate_diffusivities!(K, arch, grid, closure::AnisotropicBiharmonicDiffusivity, args...) = nothing

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::AnisotropicBiharmonicDiffusivity, U, K) = (
    - closure.νh * ∇h⁴_fca(i, j, k, grid, U.u)
    - closure.νv * ∂z⁴_aac(i, j, k, grid, U.u)
    )

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::AnisotropicBiharmonicDiffusivity, U, K) = (
    - closure.νh * ∇h⁴_cfa(i, j, k, grid, U.v)
    - closure.νv * ∂z⁴_aac(i, j, k, grid, U.v)
    )

@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::AnisotropicBiharmonicDiffusivity, U, K) = (
    - closure.νh * ∇h⁴_cca(i, j, k, grid, U.w)
    - closure.νv * ∂z⁴_aaf(i, j, k, grid, U.w)
    )

@inline function ∇_κ_∇c(i, j, k, grid, closure::AnisotropicBiharmonicDiffusivity,
                        c, ::Val{tracer_index}, args...) where tracer_index

    @inbounds κh = closure.κh[tracer_index]
    @inbounds κv = closure.κv[tracer_index]

    return (- κh * ∇h⁴_cca(i, j, k, grid, c)
            - κv * ∂z⁴_aac(i, j, k, grid, c)
           )
end
