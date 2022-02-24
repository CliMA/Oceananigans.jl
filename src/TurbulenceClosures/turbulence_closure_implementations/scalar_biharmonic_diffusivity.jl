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
    ScalarBiharmonicDiffusivity(FT=Float64; νh=0, κh=0, νz=nothing, κz=nothing)

Returns parameters for a scalar biharmonic diffusivity model.

Keyword arguments
=================

  - `νh`: Horizontal viscosity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`.

  - `νz`: Vertical viscosity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`.

  - `κh`: Horizontal diffusivity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`, or
          `NamedTuple` of diffusivities with entries for each tracer.

  - `κz`: Vertical diffusivity. `Number`, `AbstractArray`, or `Function(x, y, z, t)`, or
          `NamedTuple` of diffusivities with entries for each tracer.
"""
function ScalarBiharmonicDiffusivity(FT=Float64; ν=0, κ=0, discrete_diffusivity = false, isotropy::Iso = Horizontal()) where {Iso}
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

function Base.summary(closure::ScalarBiharmonicDiffusivity)
    Iso = summary(isotropy(closure))
    return string("ScalarBiharmonicDiffusivity{$Iso} with ν=", summary(closure.ν), " and κ=", summary(closure.κ))
end

Base.show(io::IO, closure::ScalarBiharmonicDiffusivity) = print(io, summary(closure))
    
