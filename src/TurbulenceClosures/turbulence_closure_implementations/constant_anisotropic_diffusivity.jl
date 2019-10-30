"""
    ConstantAnisotropicDiffusivity{FT, KH, KV}

Parameters for constant anisotropic diffusivity models.
"""
struct ConstantAnisotropicDiffusivity{FT, KH, KV} <: TensorDiffusivity{FT}
    νh :: FT
    νv :: FT
    κh :: KH
    κv :: KV
    function ConstantAnisotropicDiffusivity{FT}(νh, νv, κh, κv) where FT
        return new{FT, typeof(κh), typeof(κv)}(νh, νv, convert_diffusivity(FT, κh), convert_diffusivity(FT, κv))
    end
end

"""
    ConstantAnisotropicDiffusivity(; νh, νv, κh, κv)

Returns parameters for a constant anisotropic diffusivity closure with constant horizontal
and vertical viscosities `νh`, `νv` and constant horizontal and vertical thermal 
diffusivities `κh`, `κv`. 

By default, a viscosity of `ν = 1.05×10⁻⁶` m² s⁻¹ is used for both the horizontal 
and vertical viscosity, and a diffusivity of `κ = 1.46×10⁻⁷` m² s⁻¹ is used
for the horizontal and vertical diffusivities applied to every tracer.
These values are the approximate viscosity and thermal diffusivity for seawater at 20°C 
and 35 psu, according to Sharqawy et al., "Thermophysical properties of seawater: A review 
of existing correlations and data" (2010).
"""
ConstantAnisotropicDiffusivity(FT=Float64; νh=ν₀, νv=ν₀, κh=κ₀, κv=κ₀) =
    ConstantAnisotropicDiffusivity{FT}(νh, νv, κh, κv)

function with_tracers(tracers, closure::ConstantAnisotropicDiffusivity{FT}) where FT
    κh = tracer_diffusivities(tracers, closure.κh)
    κv = tracer_diffusivities(tracers, closure.κv)
    return ConstantAnisotropicDiffusivity{FT}(closure.νh, closure.νv, κh, κv)
end

calculate_diffusivities!(K, arch, grid, closure::ConstantAnisotropicDiffusivity, args...) = nothing

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, U, K) = (
      closure.νh * ∇h²_fca(i, j, k, grid, U.u)
    + closure.νv * ∂z²_aac(i, j, k, grid, U.u)
    )

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, U, K) = (
      closure.νh * ∇h²_cfa(i, j, k, grid, U.v)
    + closure.νv * ∂z²_aac(i, j, k, grid, U.v)
    )

@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, U, K) = (
      closure.νh * ∇h²_cca(i, j, k, grid, U.w)
    + closure.νv * ∂z²_aaf(i, j, k, grid, U.w)
    )

@inline function ∇_κ_∇c(i, j, k, grid, closure::ConstantAnisotropicDiffusivity,
                        c, ::Val{tracer_index}, args...) where tracer_index

    @inbounds κh = closure.κh[tracer_index]
    @inbounds κv = closure.κv[tracer_index]

    return (  κh * ∇h²_cca(i, j, k, grid, c)
            + κv * ∂z²_aac(i, j, k, grid, c)
           )
end
