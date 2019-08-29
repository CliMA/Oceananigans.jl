module TurbulenceClosures

export
  IsotropicDiffusivity,
  ConstantIsotropicDiffusivity,

  ConstantAnisotropicDiffusivity,

  ConstantSmagorinsky,
  DeardorffSmagorinsky,
  BlasiusSmagorinsky,

  AnisotropicMinimumDissipation,
  RozemaAnisotropicMinimumDissipation,
  VerstappenAnisotropicMinimumDissipation,

  TurbulentDiffusivities,
  calc_diffusivities!,

  ∇_κ_∇c,
  ∇_κ_∇T,
  ∇_κ_∇S,
  ∂ⱼ_2ν_Σ₁ⱼ,
  ∂ⱼ_2ν_Σ₂ⱼ,
  ∂ⱼ_2ν_Σ₃ⱼ

using
  Oceananigans,
  Oceananigans.Operators,
  GPUifyLoops

@hascuda using CUDAdrv, CUDAnative

abstract type TurbulenceClosure{T} end
abstract type IsotropicDiffusivity{T} <: TurbulenceClosure{T} end
abstract type TensorDiffusivity{T} <: TurbulenceClosure{T} end

@inline ∇_κ_∇T(args...) = ∇_κ_∇c(args...)
@inline ∇_κ_∇S(args...) = ∇_κ_∇c(args...)

# Approximate viscosities and thermal diffusivities for seawater
# at 20ᵒC and 35 psu, according to Sharqawy et al., "Thermophysical 
# properties of seawater: A review of existing correlations and data" (2010).
const ν₀ = 1.05e-6
const κ₀ = 1.46e-7

"""
    typed_keyword_constructor(T, Closure; kwargs...)

Return an object `Closure` with fields provided by `kwargs`
converted to type `T`. Mainly provided for converting between
different float types when working with constructors associated
with types defined via `Base.@kwdef`.
"""
function typed_keyword_constructor(T, Closure; kwargs...)
    closure = Closure(; kwargs...)
    names = fieldnames(Closure)
    vals = [getproperty(closure, name) for name in names]
    return Closure{T}(vals...)
end

Base.eltype(::TurbulenceClosure{T}) where T = T

include("closure_operators.jl")
include("velocity_tracer_gradients.jl")
include("constant_diffusivity_closures.jl")
include("smagorinsky.jl")
include("rozema_anisotropic_minimum_dissipation.jl")
include("verstappen_anisotropic_minimum_dissipation.jl")

# Some value judgements here:
const AnisotropicMinimumDissipation = VerstappenAnisotropicMinimumDissipation
const ConstantSmagorinsky = DeardorffSmagorinsky

# For easy conversion of the float type associated with a turbulence closure struct:
basetype(::ConstantSmagorinsky) = ConstantSmagorinsky
basetype(::BlasiusSmagorinsky) = BlasiusSmagorinsky
basetype(::ConstantIsotropicDiffusivity) = ConstantIsotropicDiffusivity
basetype(::AnisotropicMinimumDissipation) = AnisotropicMinimumDissipation
basetype(::RozemaAnisotropicMinimumDissipation) = RozemaAnisotropicMinimumDissipation

function Base.convert(::TurbulenceClosure{T2}, closure::TurbulenceClosure{T1}) where {T1, T2}
    paramdict = Dict((p, convert(T2, getproperty(closure, p))) for p in propertynames(closure))
    return basetype(closure)(T2; paramdict...)
end

# Fallback constructor for diffusivity types withotu precomputed diffusivities:
TurbulentDiffusivities(arch::Architecture, grid::Grid, args...) = nothing

end # module
