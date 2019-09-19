#####
##### Constant diffusivity
#####

"""
    ConstantIsotropicDiffusivity{T}

    ConstantIsotropicDiffusivity(; ν, κ)

Returns parameters for a constant isotropic diffusivity closure with constant viscosity `ν`
and constant thermal diffusivity `κ`. `ν` and `κ` may represent molecular diffusivities or
turbulent eddy diffusivities.

By default, a molecular viscosity of ``ν = 1.05×10⁻⁶`` m²/s and a molecular thermal
diffusivity of ``κ = 1.46×10⁻⁷`` m²/s is used, corresponding to the use of no turbulent
diffusivity closure at all. These molecular values are the approximate viscosity and
thermal diffusivity for seawater at 20°C and 35 psu, according to Sharqawy et al.,
"Thermophysical properties of seawater: A review of existing correlations and data" (2010).
"""
Base.@kwdef struct ConstantIsotropicDiffusivity{T} <: IsotropicDiffusivity{T}
    ν :: T = ν₀
    κ :: T = κ₀
end

ConstantIsotropicDiffusivity(T; kwargs...) =
    typed_keyword_constructor(T, ConstantIsotropicDiffusivity; kwargs...)

calc_diffusivities!(diffusivities, grid, closure::ConstantIsotropicDiffusivity,
                    args...) = nothing

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

#####
##### Constant anisotropic diffusivity (tensor diffusivity with heterogeneous
#####                                       diagonal components)
#####

"""
    ConstantAnisotropicDiffusivity{T}

    ConstantAnisotropicDiffusivity(; νh, νv, κh, κv)

Returns parameters for a constant anisotropic diffusivity closure with constant horizontal
and vertical viscosities `νh`, `νv` and constant horizontal and vertical thermal diffusivities
`κh`, `κv`. `ν` and `κ` may represent molecular diffusivities or turbulent eddy diffusivities.

By default, a molecular viscosity of ``ν = 1.05×10⁻⁶`` m²/s and a molecular thermal
diffusivity of ``κ = 1.46×10⁻⁷`` m²/s is used, corresponding to the use of no turbulent
diffusivity closure at all. These molecular values are the approximate viscosity and
thermal diffusivity for seawater at 20°C and 35 psu, according to Sharqawy et al.,
"Thermophysical properties of seawater: A review of existing correlations and data" (2010).
"""
Base.@kwdef struct ConstantAnisotropicDiffusivity{T} <: TensorDiffusivity{T}
    νh :: T = ν₀
    νv :: T = ν₀
    κh :: T = κ₀
    κv :: T = κ₀
end

calc_diffusivities!(diffusivities, grid, closure::ConstantAnisotropicDiffusivity,
                    args...) = nothing

ConstantAnisotropicDiffusivity(T; kwargs...) =
    typed_keyword_constructor(T, ConstantAnisotropicDiffusivity; kwargs...)

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
      closure.κh * ∂x²_caa(i, j, k, grid, c)
    + closure.κh * ∂y²_aca(i, j, k, grid, c)
    + closure.κv * ∂z²_aac(i, j, k, grid, c)
    )
