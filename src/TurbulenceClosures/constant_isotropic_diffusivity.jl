#####
##### Constant diffusivity
#####

"""
    ConstantIsotropicDiffusivity{FT, K}

Parameters for constant isotropic diffusivity models.
"""
struct ConstantIsotropicDiffusivity{FT, K} <: IsotropicDiffusivity{FT}
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

calculate_diffusivities!(K, grid, closure::ConstantIsotropicDiffusivity, args...) = nothing

@inline function ∇_κ_∇c(i, j, k, grid, c, tracer_idx, closure::ConstantIsotropicDiffusivity, args...)
    @inbounds κ = closure.κ[tracer_idx]

    return (  κ / grid.Δx^2 * δx²_c2f2c(grid, c, i, j, k)
            + κ / grid.Δy^2 * δy²_c2f2c(grid, c, i, j, k)
            + κ / grid.Δz^2 * δz²_c2f2c(grid, c, i, j, k)
           )
end

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, u, v, w, args...) = (
      closure.ν / grid.Δx^2 * δx²_f2c2f(grid, u, i, j, k)
    + closure.ν / grid.Δy^2 * δy²_f2e2f(grid, u, i, j, k)
    + closure.ν / grid.Δz^2 * δz²_f2e2f(grid, u, i, j, k)
)

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, u, v, w, args...) = (
      closure.ν / grid.Δx^2 * δx²_f2e2f(grid, v, i, j, k)
    + closure.ν / grid.Δy^2 * δy²_f2c2f(grid, v, i, j, k)
    + closure.ν / grid.Δz^2 * δz²_f2e2f(grid, v, i, j, k)
)

@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, u, v, w, args...) = (
      closure.ν / grid.Δx^2 * δx²_f2e2f(grid, w, i, j, k)
    + closure.ν / grid.Δy^2 * δy²_f2e2f(grid, w, i, j, k)
    + closure.ν / grid.Δz^2 * δz²_f2c2f(grid, w, i, j, k)
)
