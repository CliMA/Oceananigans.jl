#####
##### Constant diffusivity
#####

"""
    ConstantIsotropicDiffusivity{FT, K}

Parameters for constant isotropic diffusivity models.
"""
struct ConstantIsotropicDiffusivity{FT, K} <: IsotropicViscosity{FT}
    ν :: FT
    κ :: K
    function ConstantIsotropicDiffusivity{FT}(ν, κ) where FT
        return new{FT, typeof(κ)}(ν, convert_diffusivity(FT, κ))
    end
end

"""
    ConstantIsotropicDiffusivity([FT=Float64;] ν, κ)

Returns parameters for a constant isotropic diffusivity model with constant viscosity `ν`
and constant thermal diffusivities `κ` for each tracer field in `tracers`
`ν` and the fields of `κ` may represent molecular diffusivities in cases that all flow
features are explicitly resovled, or turbulent eddy diffusivities that model the effect of
unresolved, subgrid-scale turbulence.
`κ` may be a `NamedTuple` with fields corresponding
to each tracer, or a single number to be a applied to all tracers.

By default, a molecular viscosity of `ν = 1.05×10⁻⁶` m² s⁻¹ and a molecular thermal
diffusivity of `κ = 1.46×10⁻⁷` m² s⁻¹ is used for each tracer. These molecular values are
the approximate viscosity and thermal diffusivity for seawater at 20°C and 35 psu,
according to Sharqawy et al., "Thermophysical properties of seawater: A review of existing
correlations and data" (2010).
"""
ConstantIsotropicDiffusivity(FT=Float64; ν=ν₀, κ=κ₀) = ConstantIsotropicDiffusivity{FT}(ν, κ)

function with_tracers(tracers, closure::ConstantIsotropicDiffusivity{FT}) where FT
    κ = tracer_diffusivities(tracers, closure.κ)
    return ConstantIsotropicDiffusivity{FT}(closure.ν, κ)
end

calculate_diffusivities!(K, arch, grid, closure::ConstantIsotropicDiffusivity, args...) = nothing

@inline function ∇_κ_∇c(i, j, k, grid, closure::ConstantIsotropicDiffusivity,
                        c, ::Val{tracer_index}, args...) where tracer_index

    @inbounds κ = closure.κ[tracer_index]

    return div_κ∇c(i, j, k, grid, κ, c)
end

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, U, args...) = div_ν∇u(i, j, k, grid, closure.ν, U.u)
@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, U, args...) = div_ν∇v(i, j, k, grid, closure.ν, U.v)
@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, U, args...) = div_ν∇w(i, j, k, grid, closure.ν, U.w)
