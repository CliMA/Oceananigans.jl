import Oceananigans.Grids: required_halo_size

struct ScalarDiffusivity{TD, Iso, N, K} <: AbstractScalarDiffusivity{TD, Iso}
    ν :: N
    κ :: K

    function ScalarDiffusivity{TD, Iso}(ν::N, κ::K) where {TD, Iso, N, K}
        return new{TD, Iso, N, K}(ν, κ)
    end
end

"""
    ScalarDiffusivity([FT=Float64;]
                         ν=0, κ=0, time_discretization = Explicit())

Returns parameters for an isotropic diffusivity model with viscosity `ν`
and thermal diffusivities `κ` for each tracer field in `tracers`
`ν` and the fields of `κ` may be constants, arrays, fields, or
functions of `(x, y, z, t)`.

`κ` may be a `NamedTuple` with fields corresponding to each tracer, or a
single number to be a applied to all tracers.
"""
function ScalarDiffusivity(FT=Float64;
                              ν=0, κ=0,
                              discrete_diffusivity = false,
                              isotropy::Iso=ThreeDimensional(),
                              time_discretization::TD = Explicit()) where {TD, Iso}

    if isotropy == Horizontal() && time_discretization == VerticallyImplicit()
        throw(ArgumentError("VerticallyImplicitTimeDiscretization is only supported for `isotropy = Horizontal()` or `isotropy = ThreeDimensional()`"))
    end
                                  
    
        return ScalarDiffusivity{TD, Iso}(ν, κ)
    end
end

required_halo_size(closure::ScalarDiffusivity) = 1 
 
function with_tracers(tracers, closure::ScalarDiffusivity{TD, Iso}) where {TD, Iso}
    κ = tracer_diffusivities(tracers, closure.κ)
    return ScalarDiffusivity{TD, Iso}(closure.ν, κ)
end

@inline viscosity(closure::ScalarDiffusivity, args...) = closure.ν
@inline diffusivity(closure::ScalarDiffusivity, ::Val{tracer_index}, args...) where tracer_index = closure.κ[tracer_index]
                    
calculate_diffusivities!(diffusivities, ::ScalarDiffusivity, args...) = nothing

Base.show(io::IO, closure::ScalarDiffusivity{TD, Iso})  where {TD, Iso}= 
    print(io, "ScalarDiffusivity:\n",
              "ν=$(closure.ν), κ=$(closure.κ)\n",
              "time discretization: $(time_discretization(closure))\n",
              "isotropy: $Iso")
