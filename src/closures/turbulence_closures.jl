module TurbulenceClosures

export
  ConstantIsotropicDiffusivity,
  DirectionalDiffusivity,
  ConstantSmagorinsky,

  ∇_κ_∇ϕ,
  ∂ⱼ_2ν_Σ₁ⱼ,
  ∂ⱼ_2ν_Σ₂ⱼ,
  ∂ⱼ_2ν_Σ₃ⱼ,

  ν_ccc,
  ν_ffc,
  ν_fcf,
  ν_cff,
  κ_ccc

using Oceananigans, Oceananigans.Operators

using Oceananigans.Operators: incmod1, decmod1

abstract type TurbulenceClosure{T} end
abstract type ScalarDiffusivityClosure{T} <: TurbulenceClosure{T} end
abstract type TensorDiffusivityClosure{T} <: TurbulenceClosure{T} end

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
