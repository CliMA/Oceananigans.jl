import Oceananigans.Grids: required_halo_size

"""
    ScalarBiharmonicDiffusivity{N, K, Dir}

Holds viscosity and diffusivities for models with prescribed isotropic diffusivities.
"""
struct ScalarBiharmonicDiffusivity{N, K, Dir} <: AbstractScalarBiharmonicDiffusivity{Dir}
    ν :: N
    κ :: K

    function ScalarBiharmonicDiffusivity{Dir}(ν::N, κ::K) where {Dir, N, K}
        return new{Dir, N, K}(ν, κ)
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
function ScalarBiharmonicDiffusivity(FT=Float64; ν=0, κ=0, direction = :Horizontal)
    ν = convert_diffusivity(FT, ν)
    κ = convert_diffusivity(FT, κ)
    return ScalarBiharmonicDiffusivity{eval(direction)}(FT(ν), κ)
end

function with_tracers(tracers, closure::ScalarBiharmonicDiffusivity{Dir})
    κ = tracer_diffusivities(tracers, closure.κ)
    return ScalarBiharmonicDiffusivity{Dir}(closure.ν, κ)
end

calculate_diffusivities!(diffusivities, closure::ScalarBiharmonicDiffusivity, args...) = nothing

Base.show(io::IO, closure::ScalarBiharmonicDiffusivity{Dir}) where {Dir} = 
    print(io, "ScalarBiharmonicDiffusivity: " *
              "(ν=$(closure.ν), κ=$(closure.κ)), " *
              "direction: $Dir")
