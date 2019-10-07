"""
    ConstantAnisotropicDiffusivity{T}

Parameters for constant anisotropic diffusivity models.
"""
struct ConstantAnisotropicDiffusivity{T, KH, KV} <: TensorDiffusivity{T}
    νh :: T
    νv :: T
    κh :: KH
    κv :: KV
    function ConstantAnisotropicDiffusivity{T}(νh, νv, κh, κv) where T
        κh = convert_diffusivity(T, κh)
        κv = convert_diffusivity(T, κv)
        return new{T, typeof(κh), typeof(κv)}(νh, νv, κh, κv)
    end
end

"""
    ConstantAnisotropicDiffusivity(; νh, νv, κh, κv)

Returns parameters for a constant anisotropic diffusivity closure with constant horizontal
and vertical viscosities `νh`, `νv` and constant horizontal and vertical thermal 
diffusivities `κh`, `κv`. 

By default, a viscosity of ``ν = 1.05×10⁻⁶`` m² s⁻¹ is used for both the horizontal 
and vertical viscosity, and a diffusivity of ``κ = 1.46×10⁻⁷`` m² s⁻¹ is used
for the horizontal and vertical diffusivities applied to every tracer.
These values are the approximate viscosity and thermal diffusivity for seawater at 20°C 
and 35 psu, according to Sharqawy et al., "Thermophysical properties of seawater: A review 
of existing correlations and data" (2010).
"""
ConstantAnisotropicDiffusivity(T=Float64; νh=ν₀, νv=ν₀, κh=κ₀, κv=κ₀) =
    ConstantAnisotropicDiffusivity{T}(νh, νv, κh, κv)

function with_tracers(tracers, closure::ConstantAnisotropicDiffusivity{T}) where T
    κh = tracer_diffusivities(tracers, closure.κh)
    κv = tracer_diffusivities(tracers, closure.κv)
    return ConstantAnisotropicDiffusivity{T}(closure.νh, closure.νv, κh, κv)
end

calc_diffusivities!(diffusivities, grid, closure::ConstantAnisotropicDiffusivity,
                    args...) = nothing

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, u, v, w, K) = (
      closure.νh * ∂x²_faa(i, j, k, grid, u)
    + closure.νh * ∂y²_aca(i, j, k, grid, u)
    + closure.νv * ∂z²_aac(i, j, k, grid, u)
    )

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, u, v, w, K) = (
      closure.νh * ∂x²_caa(i, j, k, grid, v)
    + closure.νh * ∂y²_afa(i, j, k, grid, v)
    + closure.νv * ∂z²_aac(i, j, k, grid, v)
    )

@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, u, v, w, K) = (
      closure.νh * ∂x²_caa(i, j, k, grid, w)
    + closure.νh * ∂y²_aca(i, j, k, grid, w)
    + closure.νv * ∂z²_aaf(i, j, k, grid, w)
    )

@inline ∇_κ_∇c(i, j, k, grid, c, closure::ConstantAnisotropicDiffusivity, K) = (
      closure.κh[1] * ∂x²_caa(i, j, k, grid, c)
    + closure.κh[1] * ∂y²_aca(i, j, k, grid, c)
    + closure.κv[1] * ∂z²_aac(i, j, k, grid, c)
    )
