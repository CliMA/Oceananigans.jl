#####
##### Constant diffusivity
#####

"""
    ConstantIsotropicDiffusivity{T, N, S}

Parameters for constant isotropic diffusivity models.
"""
struct ConstantIsotropicDiffusivity{T, K} <: IsotropicDiffusivity{T}
    ν :: T
    κ :: K
    function ConstantIsotropicDiffusivity{T}(ν, κ) where T
        κ = convert_diffusivity(T, κ)
        return new{T, typeof(κ)}(ν, κ)
    end
end

"""
    ConstantIsotropicDiffusivity([T=Float64;] ν, κ)

Returns parameters for a constant isotropic diffusivity model with constant viscosity `ν`
and constant thermal diffusivities `κ` for each tracer field in `tracers` 
`ν` and the fields of `κ` may represent molecular diffusivities in cases that all flow 
features are explicitly resovled, or turbulent eddy diffusivities that model the effect of 
unresolved, subgrid-scale turbulence.

By default, a molecular viscosity of ``ν = 1.05×10⁻⁶`` m² s⁻¹ and a molecular thermal
diffusivity of ``κ = 1.46×10⁻⁷`` m² s⁻¹ is used for each tracer. These molecular values are 
the approximate viscosity and thermal diffusivity for seawater at 20°C and 35 psu, 
according to Sharqawy et al., "Thermophysical properties of seawater: A review of existing 
correlations and data" (2010).
"""
ConstantIsotropicDiffusivity(T=Float64; ν=ν₀, κ=κ₀) = ConstantIsotropicDiffusivity{T}(ν, κ)

function with_tracers(tracers, closure::ConstantIsotropicDiffusivity{T}) where T
    κ = tracer_diffusivities(tracers, closure.κ)
    return ConstantIsotropicDiffusivity{T}(closure.ν, κ)
end

calc_diffusivities!(diffusivities, grid, closure::ConstantIsotropicDiffusivity, args...) = nothing

@inline ∇_κ_∇c(i, j, k, grid, c, closure::ConstantIsotropicDiffusivity, args...) = (
      closure.κ / grid.Δx^2 * δx²_c2f2c(grid, c, i, j, k)
    + closure.κ / grid.Δy^2 * δy²_c2f2c(grid, c, i, j, k)
    + closure.κ / grid.Δz^2 * δz²_c2f2c(grid, c, i, j, k)
)

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
