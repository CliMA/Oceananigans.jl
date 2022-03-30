import Oceananigans.Grids: required_halo_size
using Oceananigans.Utils: prettysummary

"""
    struct ScalarBiharmonicDiffusivity{F, N, K} <: AbstractScalarBiharmonicDiffusivity{F}

Holds viscosity and diffusivities for models with prescribed isotropic diffusivities.
"""
struct ScalarBiharmonicDiffusivity{F, N, K} <: AbstractScalarBiharmonicDiffusivity{F}
    ν :: N
    κ :: K

    function ScalarBiharmonicDiffusivity{F}(ν::N, κ::K) where {F, N, K}
        return new{F, N, K}(ν, κ)
    end
end

# Aliases that allow specify the floating type, assuming that the discretization is Explicit in time
          ScalarBiharmonicDiffusivity(FT::DataType;         kwargs...) = ScalarBiharmonicDiffusivity(ThreeDimensionalFormulation(), FT; kwargs...)
  VerticalScalarBiharmonicDiffusivity(FT::DataType=Float64; kwargs...) = ScalarBiharmonicDiffusivity(VerticalFormulation(), FT; kwargs...)
HorizontalScalarBiharmonicDiffusivity(FT::DataType=Float64; kwargs...) = ScalarBiharmonicDiffusivity(HorizontalFormulation(), FT; kwargs...)

required_halo_size(::ScalarBiharmonicDiffusivity) = 2

"""
    ScalarBiharmonicDiffusivity([formulation=ThreeDimensionalFormulation(), FT=Float64;]
                                ν=0, κ=0,
                                discrete_form = false)

Return a scalar biharmonic diffusivity turbulence closure with viscosity coefficient `ν` and tracer
diffusivities `κ` for each tracer field in `tracers`. If a single `κ` is provided, it is applied to
all tracers. Otherwise `κ` must be a `NamedTuple` with values for every tracer individually.

Arguments
=========

* `formulation`:
  - `HorizontalFormulation()` for diffusivity applied in the horizontal direction(s)
  - `VerticalFormulation()` for diffusivity applied in the vertical direction,
  - `ThreeDimensionalFormulation()` (default) for diffusivity applied isotropically to all directions

* `FT`: the float datatype (default: `Float64`)

Keyword arguments
=================

  - `ν`: Viscosity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`.

  - `κ`: Diffusivity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`, or
         `NamedTuple` of diffusivities with entries for each tracer.

  - `discrete_form`: `Boolean`.
"""
function ScalarBiharmonicDiffusivity(formulation=ThreeDimensionalFormulation(), FT=Float64;
                                     ν=0, κ=0,
                                     discrete_form = false)

    ν = convert_diffusivity(FT, ν; discrete_form)
    κ = convert_diffusivity(FT, κ; discrete_form)
    return ScalarBiharmonicDiffusivity{typeof(formulation)}(ν, κ)
end

function with_tracers(tracers, closure::ScalarBiharmonicDiffusivity{F}) where {F}
    κ = tracer_diffusivities(tracers, closure.κ)
    return ScalarBiharmonicDiffusivity{F}(closure.ν, κ)
end

@inline viscosity(closure::ScalarBiharmonicDiffusivity, K) = closure.ν
@inline diffusivity(closure::ScalarBiharmonicDiffusivity, K, ::Val{id}) where id = closure.κ[id]

calculate_diffusivities!(diffusivities, closure::ScalarBiharmonicDiffusivity, args...) = nothing

function Base.summary(closure::ScalarBiharmonicDiffusivity)
    F = summary(formulation(closure))
    return string("ScalarBiharmonicDiffusivity{$F}(ν=", prettysummary(closure.ν), ", κ=", prettysummary(closure.κ), ")")
end

Base.show(io::IO, closure::ScalarBiharmonicDiffusivity) = print(io, summary(closure))
