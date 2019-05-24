#
# Constant diffusivity
#

"""
    ConstantIsotropicDiffusivity(T=Float64; ν=1e-6, κ=1e-7)

Return a `ConstantIsotropicDiffusivity` closure object of type `T` with
viscosity `ν` and scalar diffusivity `κ`.
"""
Base.@kwdef struct ConstantIsotropicDiffusivity{T} <: IsotropicDiffusivity{T}
    ν :: T = 1e-6
    κ :: T = 1e-7
end

const MolecularDiffusivity = ConstantIsotropicDiffusivity

ConstantIsotropicDiffusivity(T; kwargs...) =
    typed_keyword_constructor(T, ConstantIsotropicDiffusivity; kwargs...)

κ_ccc(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.κ

∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, u, v, w, T, S) = (
      closure.ν * ∂x²_faa(i, j, k, grid, u)
    + closure.ν * ∂y²_aca(i, j, k, grid, u)
    + closure.ν * ∂z²_aac(i, j, k, grid, u)
    )

∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, u, v, w, T, S) = (
      closure.ν * ∂x²_caa(i, j, k, grid, v)
    + closure.ν * ∂y²_afa(i, j, k, grid, v)
    + closure.ν * ∂z²_aac(i, j, k, grid, v)
    )

∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, u, v, w, T, S) = (
      closure.ν * ∂x²_caa(i, j, k, grid, w)
    + closure.ν * ∂y²_aca(i, j, k, grid, w)
    + closure.ν * ∂z²_aaf(i, j, k, grid, w)
    )

#
# Directional diffusivity (tensor diffusivity with heterogeneous
#                          diagonal components)
#

"""
    DirectionalDiffusivity(T=Float64; νh=1e-6, νv=1e-6, κh=1e-6, κv=1e-6)

Returns a DirectionalDiffusivity object with horizontal viscosity and
diffusivity `νh` and `κh`, and vertical viscosity and diffusivity
`νv` and `κv`.
"""
Base.@kwdef struct DirectionalDiffusivity{T} <: TensorDiffusivity{T}
    νh :: T = 1e-6
    νv :: T = 1e-6
    κh :: T = 1e-6
    κv :: T = 1e-6
end

DirectionalDiffusivity(T; kwargs...) =
    typed_keyword_constructor(T, DirectionalDiffusivity; kwargs...)

∇_κ_∇ϕ(i, j, k, grid, ϕ, closure::DirectionalDiffusivity, uvwTS...) = (
      ∂x_κ_∂x_ϕ(i, j, k, grid, ϕ, closure.κh, closure, uvwTS...)
    + ∂y_κ_∂y_ϕ(i, j, k, grid, ϕ, closure.κh, closure, uvwTS...)
    + ∂z_κ_∂z_ϕ(i, j, k, grid, ϕ, closure.κv, closure, uvwTS...)
    )

∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::DirectionalDiffusivity, u, v, w, T, S) = (
      closure.νh * ∂x²_faa(i, j, k, grid, u)
    + closure.νh * ∂y²_aca(i, j, k, grid, u)
    + closure.νv * ∂z²_aac(i, j, k, grid, u)
    )

∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::DirectionalDiffusivity, u, v, w, T, S) = (
      closure.νh * ∂x²_caa(i, j, k, grid, v)
    + closure.νh * ∂y²_afa(i, j, k, grid, v)
    + closure.νv * ∂z²_aac(i, j, k, grid, v)
    )

∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::DirectionalDiffusivity, u, v, w, T, S) = (
      closure.νh * ∂x²_caa(i, j, k, grid, w)
    + closure.νh * ∂y²_aca(i, j, k, grid, w)
    + closure.νv * ∂z²_aaf(i, j, k, grid, w)
    )
