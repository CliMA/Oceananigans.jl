import Oceananigans.Grids: required_halo_size
using Oceananigans.Utils: prettysummary

struct ScalarDiffusivity{TD, F, N, K} <: AbstractScalarDiffusivity{TD, F}
    ν :: N
    κ :: K

    function ScalarDiffusivity{TD, F}(ν::N, κ::K) where {TD, F, N, K}
        return new{TD, F, N, K}(ν, κ)
    end
end

"""
    ScalarDiffusivity([time_discretization=ExplicitTimeDiscretization,
                      formulation=ThreeDimensionalFormulation(), FT=Float64];
                      ν=0, κ=0,
                      discrete_form = false) 

Return `ScalarDiffusivity` with viscosity `ν` and tracer diffusivities `κ`
for each tracer field in `tracers`. If a single `κ` is provided, it is
applied to all tracers. Otherwise `κ` must be a `NamedTuple` with values
for every tracer individually.

`formulation`:
  - `HorizontalFormulation()` for diffusivity applied in the horizontal direction(s)
  - `VerticalFormulation()` for diffusivity applied in the vertical direction,
  - `ThreeDimensionalFormulation()` (default) for diffusivity applied isotropically to all directions

`ν` and the fields of `κ` may be constants (converted to `FT`), arrays, fields or
  - functions of `(x, y, z, t)` if `discrete_form = false`
  - functions of `(i, j, k, grid, LX, LY, LZ)` with `LX`, `LY` and `LZ` are either `Face()` or `Center()` if
    `discrete_form = true`.
"""
function ScalarDiffusivity(time_discretization=ExplicitTimeDiscretization(),
                           formulation=ThreeDimensionalFormulation(), FT=Float64;
                           ν=0, κ=0,
                           discrete_form = false)

    if formulation == HorizontalFormulation() && time_discretization == VerticallyImplicitTimeDiscretization()
        throw(ArgumentError("VerticallyImplicitTimeDiscretization is only supported for `VerticalFormulation` or `ThreeDimensionalFormulation`"))
    end

    κ = convert_diffusivity(FT, κ; discrete_form)
    ν = convert_diffusivity(FT, ν; discrete_form)

    return ScalarDiffusivity{typeof(time_discretization), typeof(formulation)}(ν, κ)
end

const VerticalScalarDiffusivity{TD} = ScalarDiffusivity{TD, VerticalFormulation} where TD
const HorizontalScalarDiffusivity{TD} = ScalarDiffusivity{TD, HorizontalFormulation} where TD

VerticalScalarDiffusivity(time_discretization=ExplicitTimeDiscretization(), FT::DataType=Float64; kwargs...) =
    ScalarDiffusivity(time_discretization, VerticalFormulation(), FT; kwargs...)

HorizontalScalarDiffusivity(time_discretization=ExplicitTimeDiscretization(), FT::DataType=Float64; kwargs...) =
    ScalarDiffusivity(time_discretization, HorizontalFormulation(), FT; kwargs...)

# Aliases that allow specify the floating type, assuming that the discretization is Explicit in time
          ScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization(), ThreeDimensionalFormulation(), FT; kwargs...)
  VerticalScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization(), VerticalFormulation(), FT; kwargs...)
HorizontalScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization(), HorizontalFormulation(), FT; kwargs...)

required_halo_size(closure::ScalarDiffusivity) = 1 
 
function with_tracers(tracers, closure::ScalarDiffusivity{TD, F}) where {TD, F}
    κ = tracer_diffusivities(tracers, closure.κ)
    return ScalarDiffusivity{TD, F}(closure.ν, κ)
end

@inline viscosity(closure::ScalarDiffusivity, args...) = closure.ν
@inline diffusivity(closure::ScalarDiffusivity, ::Val{tracer_index}, args...) where tracer_index = closure.κ[tracer_index]

# z_viscosity and z_diffusivity fallback to viscosity and diffusivity
# for both ThreeDimensional and VerticalScalarDiffusivity
@inline z_viscosity(closure::HorizontalScalarDiffusivity, args...) = 0
@inline z_diffusivity(closure::HorizontalScalarDiffusivity, args...) = 0

calculate_diffusivities!(diffusivities, ::ScalarDiffusivity, args...) = nothing

# Note: we could compute ν and κ (if they are Field):
# function calculate_diffusivities!(diffusivities, closure::ScalarDiffusivity, args...)
#     compute!(viscosity(closure, diffusivities))
#     !isnothing(closure.κ) && Tuple(compute!(diffusivity(closure, Val(c), diffusivities) for c=1:length(closure.κ)))
#     return nothing
# end

function Base.summary(closure::ScalarDiffusivity)
    TD = summary(time_discretization(closure))
    prefix = replace(summary(formulation(closure)), "Formulation" => "")
    prefix === "ThreeDimensional" && (prefix = "")
    return string(prefix, "ScalarDiffusivity{$TD}(ν=", prettysummary(closure.ν), ", κ=", prettysummary(closure.κ), ")")
end

Base.show(io::IO, closure::ScalarDiffusivity) = print(io, summary(closure))
