module TurbulenceClosures

export
  MolecularDiffusivity,
  ConstantIsotropicDiffusivity,
  ConstantAnisotropicDiffusivity,
  ConstantSmagorinsky,

  ∇_κ_∇ϕ,
  ∂ⱼ_2ν_Σ₁ⱼ,
  ∂ⱼ_2ν_Σ₂ⱼ,
  ∂ⱼ_2ν_Σ₃ⱼ,

  ν_ccc, ν_ffc, ν_fcf, ν_cff,
  
  κ_ccc,

  ∂x_caa, ∂x_faa, ∂x²_caa, ∂x²_faa,
  ∂y_aca, ∂y_afa, ∂y²_aca, ∂y²_afa,
  ∂z_aac, ∂z_aaf, ∂z²_aac, ∂z²_aaf,

  ▶x_caa, ▶x_faa,
  ▶y_aca, ▶y_afa,
  ▶z_aac, ▶z_aaf,
  
  ▶xy_cca, ▶xz_cac, ▶yz_acc,
  ▶xy_cfa, ▶xz_caf, ▶yz_acf,
  ▶xy_fca, ▶xz_fac, ▶yz_afc,
  ▶xy_ffa, ▶xz_faf, ▶yz_aff

using Oceananigans, Oceananigans.Operators

using Oceananigans.Operators: incmod1, decmod1

abstract type TurbulenceClosure{T} end
abstract type IsotropicDiffusivity{T} <: TurbulenceClosure{T} end
abstract type TensorDiffusivity{T} <: TurbulenceClosure{T} end

geo_mean_Δ(grid::RegularCartesianGrid) = (grid.Δx * grid.Δy * grid.Δz)^(1/3)

function typed_keyword_constructor(T, Closure; kwargs...)
    closure = Closure(; kwargs...)
    names = fieldnames(Closure)
    vals = [getproperty(closure, name) for name in names]
    return Closure{T}(vals...)
end

Base.eltype(::TurbulenceClosure{T}) where T = T

include("closure_operators.jl")
include("velocity_gradients.jl")
include("constant_diffusivity_closures.jl")
include("constant_smagorinsky.jl")

end # module
