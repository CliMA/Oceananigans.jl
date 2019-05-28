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
@inline κ_ccc(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.κ
@inline ν_ccc(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.ν
@inline ν_ffc(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.ν
@inline ν_fcf(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.ν
@inline ν_cff(i, j, k, grid, closure::ConstantIsotropicDiffusivity, args...) = closure.ν

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.ν * ∂x²_faa(i, j, k, grid, u)
    + closure.ν * ∂y²_aca(i, j, k, grid, u)
    + closure.ν * ∂z²_aac(i, j, k, grid, u)
    )

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.ν * ∂x²_caa(i, j, k, grid, v)
    + closure.ν * ∂y²_afa(i, j, k, grid, v)
    + closure.ν * ∂z²_aac(i, j, k, grid, v)
    )

@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantIsotropicDiffusivity, eos, g, u, v, w, T, S) = (
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

#=
@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.νh/grid.Δx^2 * δx²_f2c2f(grid, u, i, j, k) 
    + closure.νh/grid.Δy^2 * δy²_f2e2f(grid, u, i, j, k) 
    + closure.νv/grid.Δz^2 * δz²_f2e2f(grid, u, i, j, k)
)

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.νh/grid.Δx^2 * δx²_f2c2f(grid, v, i, j, k) 
    + closure.νh/grid.Δy^2 * δy²_f2e2f(grid, v, i, j, k) 
    + closure.νv/grid.Δz^2 * δz²_f2e2f(grid, v, i, j, k)
)

@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.νh/grid.Δx^2 * δx²_f2c2f(grid, w, i, j, k) 
    + closure.νh/grid.Δy^2 * δy²_f2e2f(grid, w, i, j, k) 
    + closure.νv/grid.Δz^2 * δz²_f2e2f(grid, w, i, j, k)
)

@inline ∇_κ_∇ϕ(i, j, k, g, Q, closure::ConstantAnisotropicDiffusivity, args...) = (
      closure.κh/g.Δx^2 * δx²_c2f2c(g, Q, i, j, k) 
    + closure.κh/g.Δy^2 * δy²_c2f2c(g, Q, i, j, k) 
    + closure.κv/g.Δz^2 * δz²_c2f2c(g, Q, i, j, k)
)
=#

@inline ∇_κ_∇ϕ(i, j, k, grid, ϕ, closure::ConstantAnisotropicDiffusivity, args...) = (
      ∂x_κ_∂x_ϕ(i, j, k, grid, ϕ, closure.κh, closure, args...)
    + ∂y_κ_∂y_ϕ(i, j, k, grid, ϕ, closure.κh, closure, args...)
    + ∂z_κ_∂z_ϕ(i, j, k, grid, ϕ, closure.κv, closure, args...)
    )

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.νh * ∂x²_faa(i, j, k, grid, u)
    + closure.νh * ∂y²_aca(i, j, k, grid, u)
    + closure.νv * ∂z²_aac(i, j, k, grid, u)
    )

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.νh * ∂x²_caa(i, j, k, grid, v)
    + closure.νh * ∂y²_afa(i, j, k, grid, v)
    + closure.νv * ∂z²_aac(i, j, k, grid, v)
    )

@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, eos, g, u, v, w, T, S) = (
      closure.νh * ∂x²_caa(i, j, k, grid, w)
    + closure.νh * ∂y²_aca(i, j, k, grid, w)
    + closure.νv * ∂z²_aaf(i, j, k, grid, w)
    )

# These functions are used to specify Gradient and Value boundary conditions.
@inline κ₁₁_ccc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.κh
@inline κ₂₂_ccc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.κh
@inline κ₃₃_ccc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.κv

@inline ν₁₁_ccc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh
@inline ν₁₁_ffc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh
@inline ν₁₁_fcf(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh
@inline ν₁₁_cff(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh

@inline ν₂₂_ccc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh
@inline ν₂₂_ffc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh
@inline ν₂₂_fcf(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh
@inline ν₂₂_cff(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νh

@inline ν₃₃_ccc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νv
@inline ν₃₃_ffc(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νv
@inline ν₃₃_fcf(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νv
@inline ν₃₃_cff(i, j, k, grid, closure::ConstantAnisotropicDiffusivity, args...) = closure.νv
