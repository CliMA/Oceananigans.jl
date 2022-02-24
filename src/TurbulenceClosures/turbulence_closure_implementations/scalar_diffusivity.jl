import Oceananigans.Grids: required_halo_size

struct ScalarDiffusivity{TD, F, N, K} <: AbstractScalarDiffusivity{TD, F}
    ν :: N
    κ :: K

    function ScalarDiffusivity{TD, F}(ν::N, κ::K) where {TD, F, N, K}
        return new{TD, F, N, K}(ν, κ)
    end
end

"""
ScalarDiffusivity(time_disc=ExplicitTimeDiscretization,
                 formulation=ThreeDimensionalFormulation, FT=Float64;
                 ν=0, κ=0,
                 discrete_form = false) 

Returns parameters for an isotropic diffusivity model with viscosity `ν`
and thermal diffusivities `κ` for each tracer field in `tracers`
`ν` and the fields of `κ` may be constants, arrays, fields or
- functions of `(x, y, z, t)` if `discrete_diffusivity = false`
- functions of `(i, j, k, grid, LX, LY, LZ)` with `LX`, `LY` and `LZ` are either `Face()` or `Center()` if
  `discrete_form = true`.

`κ` may be a `NamedTuple` with fields corresponding to each tracer, or a
single number to be a applied to all tracers.
"""

  VerticalScalarDiffusivity(time_disc=ExplicitTimeDiscretization, FT::DataType=Float64; kwargs...) = ScalarDiffusivity(time_disc, VerticalFormulation, FT; kwargs...)
HorizontalScalarDiffusivity(time_disc=ExplicitTimeDiscretization, FT::DataType=Float64; kwargs...) = ScalarDiffusivity(time_disc, HorizontalFormulation, FT; kwargs...)

# Aliases that allow specify the floating type, assuming that the discretization is Explicit in time
          ScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization, ThreeDimensionalFormulation, FT; kwargs...)
  VerticalScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization, VerticalFormulation, FT; kwargs...)
HorizontalScalarDiffusivity(FT::DataType; kwargs...) = ScalarDiffusivity(ExplicitTimeDiscretization, HorizontalFormulation, FT; kwargs...)

function ScalarDiffusivity(time_disc=ExplicitTimeDiscretization,
                           formulation=ThreeDimensionalFormulation, FT=Float64;
                           ν=0, κ=0,
                           discrete_form = false) 

    if formulation == HorizontalFormulation && time_discretization == VerticallyImplicitTimeDiscretization
        throw(ArgumentError("VerticallyImplicitTimeDiscretization is only supported for `HorizontalFormulation` or `ThreeDimensionalFormulation`"))
    end
    κ = convert_diffusivity(FT, κ, Val(discrete_form))
    ν = convert_diffusivity(FT, ν, Val(discrete_form))
    return ScalarDiffusivity{time_disc, formulation}(ν, κ)
end

required_halo_size(closure::ScalarDiffusivity) = 1 
 
function with_tracers(tracers, closure::ScalarDiffusivity{TD, F}) where {TD, F}
    κ = tracer_diffusivities(tracers, closure.κ)
    return ScalarDiffusivity{TD, F}(closure.ν, κ)
end

@inline viscosity(closure::ScalarDiffusivity, args...) = closure.ν
@inline diffusivity(closure::ScalarDiffusivity, ::Val{tracer_index}, args...) where tracer_index = closure.κ[tracer_index]
                    
calculate_diffusivities!(diffusivities, ::ScalarDiffusivity, args...) = nothing

Base.show(io::IO, closure::ScalarDiffusivity{TD, F})  where {TD, F}= 
    print(io, "ScalarDiffusivity:\n",
              "ν=$(closure.ν), κ=$(closure.κ)\n",
              "time discretization: $TD\n",
              "formulation: $F")
