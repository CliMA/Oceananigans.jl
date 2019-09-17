#####
##### Constant diffusivity
#####

"""
    ConstantIsotropicDiffusivity(T=Float64; ν=1e-6, κ=1e-7)

or

    MolecularDiffusivity(T=Float64; ν=1e-6, κ=1e-7)

Return a `ConstantIsotropicDiffusivity` closure object of type `T` with
viscosity `ν` and scalar diffusivity `κ`.
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
    ConstantAnisotropicDiffusivity(T=Float64; νh=1e-6, νv=1e-6, κh=1e-6, κv=1e-6)

Returns a ConstantAnisotropicDiffusivity object with horizontal viscosity and
diffusivity `νh` and `κh`, and vertical viscosity and diffusivity
`νv` and `κv`.
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
