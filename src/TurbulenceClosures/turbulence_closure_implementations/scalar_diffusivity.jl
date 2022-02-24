import Oceananigans.Grids: required_halo_size
using Oceananigans.Utils: prettysummary

struct ScalarDiffusivity{TD, Iso, N, K} <: AbstractScalarDiffusivity{TD, Iso}
    ν :: N
    κ :: K

    function ScalarDiffusivity{TD, Iso}(ν::N, κ::K) where {TD, Iso, N, K}
        return new{TD, Iso, N, K}(ν, κ)
    end
end

"""
    ScalarDiffusivity([FT=Float64];
                      ν = 0,
                      κ = 0,
                      discrete_form = false,
                      isotropy = ThreeDimensional() 
                      time_discretization = Explicit())

Return `ScalarDiffusivity` with viscosity `ν` and tracer diffusivities `κ`
for each tracer field in `tracers`. If a single `κ` is provided, it is
applied to all tracers. Otherwise `κ` must be a `NamedTuple` with values
for every tracer individually. `ν` and `κ` are converted to the floating
point type `FT` is they are Number.

`ν` and `κ` (or its elements) may be constants, arrays, fields or

  - functions of `(x, y, z, t)` if `discrete_form = false`
  - functions of `(LX, LY, LZ, i, j, k, grid, t)` with `LX`, `LY` and `LZ` are either `Face()` or `Center()` if
    `discrete_form = true`.
"""
function ScalarDiffusivity(FT=Float64;
                           ν=0, κ=0,
                           discrete_form = false,
                           isotropy::Iso = ThreeDimensional(),
                           time_discretization::TD = Explicit()) where {TD, Iso}

    if isotropy == Horizontal() && time_discretization == VerticallyImplicit()
        throw(ArgumentError("VerticallyImplicitTimeDiscretization is only supported for `isotropy = Horizontal()` or `isotropy = ThreeDimensional()`"))
    end
    κ = convert_diffusivity(FT, κ, Val(discrete_form))
    ν = convert_diffusivity(FT, ν, Val(discrete_form))
    return ScalarDiffusivity{TD, Iso}(ν, κ)
end

required_halo_size(closure::ScalarDiffusivity) = 1 
 
function with_tracers(tracers, closure::ScalarDiffusivity{TD, Iso}) where {TD, Iso}
    κ = tracer_diffusivities(tracers, closure.κ)
    return ScalarDiffusivity{TD, Iso}(closure.ν, κ)
end

@inline viscosity(closure::ScalarDiffusivity, args...) = closure.ν
@inline diffusivity(closure::ScalarDiffusivity, ::Val{tracer_index}, args...) where tracer_index = closure.κ[tracer_index]
calculate_diffusivities!(diffusivities, ::ScalarDiffusivity, args...) = nothing

function Base.summary(closure::ScalarDiffusivity)
    TD = summary(time_discretization(closure))
    Iso = summary(isotropy(closure))
    return string("ScalarDiffusivity{$TD, $Iso}(ν=", prettysummary(closure.ν), ", κ=", prettysummary(closure.κ), ")")
end

Base.show(io::IO, closure::ScalarDiffusivity) = print(io, summary(closure))
    
