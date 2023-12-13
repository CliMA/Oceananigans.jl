import Adapt: adapt_structure
import Oceananigans.Grids: required_halo_size
using Oceananigans.Utils: prettysummary

struct ScalarDiffusivity{TD, F, V, K, N} <: AbstractScalarDiffusivity{TD, F, N}
    ν :: V
    κ :: K

    ScalarDiffusivity{TD, F, N}(ν::V, κ::K) where {TD, F, V, K, N} = new{TD, F, V, K, N}(ν, κ)
end

"""
    ScalarDiffusivity(time_discretization = ExplicitTimeDiscretization(),
                      formulation = ThreeDimensionalFormulation(), FT = Float64;
                      ν = 0,
                      κ = 0,
                      discrete_form = false,
                      loc = (nothing, nothing, nothing),
                      parameters = nothing)

Return `ScalarDiffusivity` turbulence closure with viscosity `ν` and tracer diffusivities `κ`
for each tracer field in `tracers`. If a single `κ` is provided, it is applied to all tracers.
Otherwise `κ` must be a `NamedTuple` with values for every tracer individually.

Arguments
=========

* `time_discretization`: either `ExplicitTimeDiscretization()` (default) or `VerticallyImplicitTimeDiscretization()`.

* `formulation`:
  - `HorizontalFormulation()` for diffusivity applied in the horizontal direction(s)
  - `VerticalFormulation()` for diffusivity applied in the vertical direction,
  - `ThreeDimensionalFormulation()` (default) for diffusivity applied isotropically to all directions

* `FT`: the float datatype (default: `Float64`)

Keyword arguments
=================

* `ν`: Viscosity. `Number`, three-dimensional `AbstractArray`, `Field`, or `Function`.

* `κ`: Diffusivity. `Number`, `AbstractArray`, `Field`, `Function`, or
       `NamedTuple` of diffusivities with entries for each tracer.

* `discrete_form`: `Boolean`; default: `False`.

When prescribing the viscosities or diffusivities as functions, depending on the value of keyword argument
`discrete_form`, the constructor expects:

  - `discrete_form = false` (default): functions of the grid's native coordinates and time, e.g., `(x, y, z, t)` for
    a `RectilinearGrid` or `(λ, φ, z, t)` for a `LatitudeLongitudeGrid`.

  - `discrete_form = true`: 
    - with `loc = (nothing, nothing, nothing)` (default): functions of `(i, j, k, grid, ℓx, ℓy, ℓz)` with `ℓx`, `ℓy` and `ℓz` either `Face()` or `Center()`.
    - with `loc = (ℓx, ℓy, ℓz)` with `ℓx`, `ℓy` and `ℓz` either `Face()` or `Center()`: functions of `(i, j, k, grid)`.

  - `parameters`: `NamedTuple` with parameters used by the functions that compute viscosity and/or diffusivity; default: `nothing`.


Examples
========

```jldoctest ScalarDiffusivity
julia> using Oceananigans

julia> ScalarDiffusivity(ν = 1000, κ=2000)
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=1000.0, κ=2000.0)
```

```jldoctest ScalarDiffusivity
julia> const depth_scale = 100;

julia> @inline ν(x, y, z) = 1000 * exp(z / depth_scale)
ν (generic function with 1 method)

julia> ScalarDiffusivity(ν = ν)
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=ν (generic function with 1 method), κ=0.0)
```

```jldoctest ScalarDiffusivity
julia> using Oceananigans.Grids: znode

julia> @inline function κ(i, j, k, grid, ℓx, ℓy, ℓz)
           z = znode(i, j, k, grid, ℓx, ℓy, ℓz)
           return 2000 * exp(z / depth_scale)
       end
κ (generic function with 1 method)

julia> ScalarDiffusivity(κ = κ, discrete_form = true)
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=0.0, κ=Oceananigans.TurbulenceClosures.DiscreteDiffusionFunction{Nothing, Nothing, Nothing, Nothing, typeof(κ)})
```

```jldoctest ScalarDiffusivity
julia> @inline function another_κ(i, j, k, grid, p)
           z = znode(i, j, k, grid)
           return 2000 * exp(z / p.depth_scale)
       end
another_κ (generic function with 1 method)

julia> ScalarDiffusivity(κ = another_κ, discrete_form = true, loc = (Center, Center, Face), parameters = (; depth_scale = 120.0))
ScalarDiffusivity{ExplicitTimeDiscretization}(ν=0.0, κ=Oceananigans.TurbulenceClosures.DiscreteDiffusionFunction{Center, Center, Face, NamedTuple{(:depth_scale,), Tuple{Float64}}, typeof(another_κ)})
```
"""
function ScalarDiffusivity(time_discretization=ExplicitTimeDiscretization(),
                           formulation=ThreeDimensionalFormulation(), FT=Float64;
                           ν=0, κ=0,
                           discrete_form = false,
                           loc = (nothing, nothing, nothing),
                           parameters = nothing,
                           required_halo_size = 1)

    if formulation == HorizontalFormulation() && time_discretization == VerticallyImplicitTimeDiscretization()
        throw(ArgumentError("VerticallyImplicitTimeDiscretization is only supported for `VerticalFormulation` or `ThreeDimensionalFormulation`"))
    end

    κ = convert_diffusivity(FT, κ; discrete_form, loc, parameters)
    ν = convert_diffusivity(FT, ν; discrete_form, loc, parameters)

    return ScalarDiffusivity{typeof(time_discretization), typeof(formulation), required_halo_size}(ν, κ)
end

# Explicit default
ScalarDiffusivity(formulation::AbstractDiffusivityFormulation, FT=Float64; kw...) =
    ScalarDiffusivity(ExplicitTimeDiscretization(), formulation, FT; kw...)

const VerticalScalarDiffusivity{TD} = ScalarDiffusivity{TD, VerticalFormulation} where TD
const HorizontalScalarDiffusivity{TD} = ScalarDiffusivity{TD, HorizontalFormulation} where TD
const HorizontalDivergenceScalarDiffusivity{TD} = ScalarDiffusivity{TD, HorizontalDivergenceFormulation} where TD

"""
    VerticalScalarDiffusivity([time_discretization=ExplicitTimeDiscretization(),
                              FT::DataType=Float64;]
                              kwargs...)

Shorthand for a `ScalarDiffusivity` with `VerticalFormulation()`. See [`ScalarDiffusivity`](@ref).
"""
VerticalScalarDiffusivity(time_discretization=ExplicitTimeDiscretization(), FT::DataType=Float64; kwargs...) =
    ScalarDiffusivity(time_discretization, VerticalFormulation(), FT; kwargs...)

"""
    HorizontalScalarDiffusivity([time_discretization=ExplicitTimeDiscretization(),
                                FT::DataType=Float64;]
                                kwargs...)

Shorthand for a `ScalarDiffusivity` with `HorizontalFormulation()`. See [`ScalarDiffusivity`](@ref).
"""
HorizontalScalarDiffusivity(time_discretization=ExplicitTimeDiscretization(), FT::DataType=Float64; kwargs...) =
    ScalarDiffusivity(time_discretization, HorizontalFormulation(), FT; kwargs...)
    
"""
    HorizontalDivergenceScalarDiffusivity([time_discretization=ExplicitTimeDiscretization(),
                                          FT::DataType=Float64;]
                                          kwargs...)

Shorthand for a `ScalarDiffusivity` with `HorizontalDivergenceFormulation()`. See [`ScalarDiffusivity`](@ref).
"""
HorizontalDivergenceScalarDiffusivity(time_discretization=ExplicitTimeDiscretization(), FT::DataType=Float64; kwargs...) =
    ScalarDiffusivity(time_discretization, HorizontalDivergenceFormulation(), FT; kwargs...)

# Aliases that allow specify the floating type, assuming that the discretization is Explicit in time
                    ScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization(), ThreeDimensionalFormulation(), FT; kwargs...)
            VerticalScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization(), VerticalFormulation(), FT; kwargs...)
          HorizontalScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization(), HorizontalFormulation(), FT; kwargs...)
HorizontalDivergenceScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization(), HorizontalDivergenceFormulation(), FT; kwargs...)

required_halo_size(closure::ScalarDiffusivity) = 1 
 
function with_tracers(tracers, closure::ScalarDiffusivity{TD, F, N}) where {TD, F, N}
    κ = tracer_diffusivities(tracers, closure.κ)
    return ScalarDiffusivity{TD, F, N}(closure.ν, κ)
end

@inline viscosity(closure::ScalarDiffusivity, K) = closure.ν
@inline diffusivity(closure::ScalarDiffusivity, K, ::Val{id}) where id = closure.κ[id]

compute_diffusivities!(diffusivities, ::ScalarDiffusivity, args...) = nothing

# Note: we could compute ν and κ (if they are Field):
# function compute_diffusivities!(diffusivities, closure::ScalarDiffusivity, args...)
#     compute!(viscosity(closure, diffusivities))
#     !isnothing(closure.κ) && Tuple(compute!(diffusivity(closure, Val(c), diffusivities) for c=1:length(closure.κ)))
#     return nothing
# end

function Base.summary(closure::ScalarDiffusivity)
    TD = summary(time_discretization(closure))
    prefix = replace(summary(formulation(closure)), "Formulation" => "")
    prefix === "ThreeDimensional" && (prefix = "")

    if closure.κ == NamedTuple()
        summary_str = string(prefix, "ScalarDiffusivity{$TD}(ν=", prettysummary(closure.ν), ")")
    else
        summary_str = string(prefix, "ScalarDiffusivity{$TD}(ν=", prettysummary(closure.ν), ", κ=", prettysummary(closure.κ), ")")
    end

    return summary_str
end

Base.show(io::IO, closure::ScalarDiffusivity) = print(io, summary(closure))

adapt_structure(to, closure::ScalarDiffusivity) = ScalarDiffusivity(adapt(closure.ν),
                                                                    adapt(closure.κ))
