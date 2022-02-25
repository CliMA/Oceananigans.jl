import Oceananigans.Grids: required_halo_size

"""
    ScalarBiharmonicDiffusivity{F, N, K}

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
    ScalarBiharmonicDiffusivity(FT=Float64; νh=0, κh=0, νz=nothing, κz=nothing)

Returns parameters for a scalar biharmonic diffusivity model.

Keyword arguments
=================

  - `ν`: Viscosity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`.

  - `κ`: Diffusivity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`, or
         `NamedTuple` of diffusivities with entries for each tracer.

  - `discrete_form`: `Boolean`.

  - `formulation`: formulation used for the discretization of the diffusivity operator.
                   Options are `VerticalFormulation`, `HorizontalFormulation` and 
                   `ThreeDimensionalFormulation`.

"""
function ScalarBiharmonicDiffusivity(formulation=ThreeDimensionalFormulation(), FT=Float64;
                                     ν=0, κ=0,
                                     discrete_form = false) 

    ν = convert_diffusivity(FT, ν, Val(discrete_form))
    κ = convert_diffusivity(FT, κ, Val(discrete_form))
    return ScalarBiharmonicDiffusivity{typeof(formulation)}(ν, κ)
end

function with_tracers(tracers, closure::ScalarBiharmonicDiffusivity{F}) where {F}
    κ = tracer_diffusivities(tracers, closure.κ)
    return ScalarBiharmonicDiffusivity{F}(closure.ν, κ)
end

@inline viscosity(closure::ScalarBiharmonicDiffusivity, args...) = closure.ν
@inline diffusivity(closure::ScalarBiharmonicDiffusivity, ::Val{tracer_index}, args...) where tracer_index = closure.κ[tracer_index]

calculate_diffusivities!(diffusivities, closure::ScalarBiharmonicDiffusivity, args...) = nothing

Base.show(io::IO, closure::ScalarBiharmonicDiffusivity{F}) where {F} = 
    print(io, "ScalarBiharmonicDiffusivity: " *
              "(ν=$(closure.ν), κ=$(closure.κ)), " *
              "formulation: $F")
