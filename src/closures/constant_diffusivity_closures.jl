#
# Constant diffusivity
#

"""
    ConstantIsotropicDiffusivity(T=Float64; ν=1e-6, κ=1e-7)

or

    MolecularDiffusivity(T=Float64; ν=1e-6, κ=1e-7)

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

# These functions are used to specify Gradient and Value boundary conditions.
κ_ccc(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.κ
ν_ccc(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.ν
ν_ffc(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.ν
ν_fcf(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.ν
ν_cff(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.ν

∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.ν * ∂x²_faa(i, j, k, grid, u)
    + closure.ν * ∂y²_aca(i, j, k, grid, u)
    + closure.ν * ∂z²_aac(i, j, k, grid, u)
    )

∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.ν * ∂x²_caa(i, j, k, grid, v)
    + closure.ν * ∂y²_afa(i, j, k, grid, v)
    + closure.ν * ∂z²_aac(i, j, k, grid, v)
    )

∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.ν * ∂x²_caa(i, j, k, grid, w)
    + closure.ν * ∂y²_aca(i, j, k, grid, w)
    + closure.ν * ∂z²_aaf(i, j, k, grid, w)
    )

#
# Constant anisotropic diffusivity (tensor diffusivity with heterogeneous
#                                       diagonal components)
#

"""
    ConstantAnisotropicDiffusivity(T=Float64; νh=1e-6, νv=1e-6, κh=1e-6, κv=1e-6)

Returns a ConstantAnisotropicDiffusivity object with horizontal viscosity and
diffusivity `νh` and `κh`, and vertical viscosity and diffusivity
`νv` and `κv`.
"""
Base.@kwdef struct ConstantAnisotropicDiffusivity{T} <: TensorDiffusivity{T}
    νh :: T = 1e-6
    νv :: T = 1e-6
    κh :: T = 1e-6
    κv :: T = 1e-6
end

ConstantAnisotropicDiffusivity(T; kwargs...) =
    typed_keyword_constructor(T, ConstantAnisotropicDiffusivity; kwargs...)

∇_κ_∇ϕ(i, j, k, grid, ϕ, closure::ConstantAnisotropicDiffusivity, args...) = (
      ∂x_κ_∂x_ϕ(i, j, k, grid, ϕ, closure.κh, closure, args...)
    + ∂y_κ_∂y_ϕ(i, j, k, grid, ϕ, closure.κh, closure, args...)
    + ∂z_κ_∂z_ϕ(i, j, k, grid, ϕ, closure.κv, closure, args...)
    )

∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.νh * ∂x²_faa(i, j, k, grid, u)
    + closure.νh * ∂y²_aca(i, j, k, grid, u)
    + closure.νv * ∂z²_aac(i, j, k, grid, u)
    )

∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.νh * ∂x²_caa(i, j, k, grid, v)
    + closure.νh * ∂y²_afa(i, j, k, grid, v)
    + closure.νv * ∂z²_aac(i, j, k, grid, v)
    )

∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.νh * ∂x²_caa(i, j, k, grid, w)
    + closure.νh * ∂y²_aca(i, j, k, grid, w)
    + closure.νv * ∂z²_aaf(i, j, k, grid, w)
    )

# These functions are used to specify Gradient and Value boundary conditions.
κ₁₁_ccc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.κh
κ₂₂_ccc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.κh
κ₃₃_ccc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.κv

ν₁₁_ccc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh
ν₁₁_ffc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh
ν₁₁_fcf(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh
ν₁₁_cff(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh

ν₂₂_ccc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh
ν₂₂_ffc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh
ν₂₂_fcf(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh
ν₂₂_cff(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh

ν₃₃_ccc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νv
ν₃₃_ffc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νv
ν₃₃_fcf(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νv
ν₃₃_cff(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νv
