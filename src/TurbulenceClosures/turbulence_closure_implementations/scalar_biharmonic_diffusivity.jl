import Oceananigans.Grids: required_halo_size

"""
    ScalarBiharmonicDiffusivity{Iso, N, K}

Holds viscosity and diffusivities for models with prescribed isotropic diffusivities.
"""
struct ScalarBiharmonicDiffusivity{Iso, N, K} <: AbstractScalarBiharmonicDiffusivity{Iso}
    ν :: N
    κ :: K

    function ScalarBiharmonicDiffusivity{Iso}(ν::N, κ::K) where {Iso, N, K}
        return new{Iso, N, K}(ν, κ)
    end
end

required_halo_size(::ScalarBiharmonicDiffusivity) = 2

"""
    ScalarBiharmonicDiffusivity(FT=Float64; ν=0, κ=0, discrete_diffusivity = false, isotropy::Iso = XYDirections())

Returns parameters for a scalar biharmonic diffusivity model.

Keyword arguments
=================

  - `ν`: Viscosity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`.

  - `κ`: Tracer diffusivity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`, or
          `NamedTuple` of diffusivities with entries for each tracer.

  - `discrete_diffusivity`: `Boolean`.

  - `isotropy`: Directions over which to apply diffusivity operator. Options are 
          `XYDirections()`, `ZDirection()` and `XYZDirections()`.

"""
function ScalarBiharmonicDiffusivity(FT=Float64; ν=0, κ=0, discrete_diffusivity = false, isotropy::Iso = XYDirections()) where {Iso}
    ν = convert_diffusivity(FT, ν, Val(discrete_diffusivity))
    κ = convert_diffusivity(FT, κ, Val(discrete_diffusivity))
    return ScalarBiharmonicDiffusivity{Iso}(ν, κ)
end

function with_tracers(tracers, closure::ScalarBiharmonicDiffusivity{Iso}) where {Iso}
    κ = tracer_diffusivities(tracers, closure.κ)
    return ScalarBiharmonicDiffusivity{Iso}(closure.ν, κ)
end

@inline viscosity(closure::ScalarBiharmonicDiffusivity, args...) = closure.ν
@inline diffusivity(closure::ScalarBiharmonicDiffusivity, ::Val{tracer_index}, args...) where tracer_index = closure.κ[tracer_index]

calculate_diffusivities!(diffusivities, closure::ScalarBiharmonicDiffusivity, args...) = nothing

Base.show(io::IO, closure::ScalarBiharmonicDiffusivity{Iso}) where {Iso} = 
    print(io, "ScalarBiharmonicDiffusivity: " *
              "(ν=$(closure.ν), κ=$(closure.κ)), " *
              "isotropy: $Iso")
