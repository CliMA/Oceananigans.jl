import Oceananigans.Grids: required_halo_size

struct ScalarDiffusivity{TD, Dir, N, K} <: AbstractScalarDiffusivity{TD, Dir}
    ν :: N
    κ :: K

    function ScalarDiffusivity{TD, Dir}(ν::N, κ::K) where {TD, Dir, N, K}
        return new{TD, Dir, N, K}(ν, κ)
    end
end

"""
    ScalarDiffusivity([FT=Float64;]
                         ν=0, κ=0, time_discretization = ExplicitTimeDiscretization())

Returns parameters for an isotropic diffusivity model with viscosity `ν`
and thermal diffusivities `κ` for each tracer field in `tracers`
`ν` and the fields of `κ` may be constants, arrays, fields, or
functions of `(x, y, z, t)`.

`κ` may be a `NamedTuple` with fields corresponding to each tracer, or a
single number to be a applied to all tracers.
"""
function ScalarDiffusivity(FT=Float64;
                              ν=0, κ=0, direction=:ThreeDimensional, time_discretization::TD = ExplicitTimeDiscretization()) where {TD, Dir}

    if ν isa Number && κ isa Number
        κ = convert_diffusivity(FT, κ)
        return ScalarDiffusivity{TD, eval(direction)}(FT(ν), κ)
    else
        return ScalarDiffusivity{TD, eval(direction)}(ν, κ)
    end
end

required_halo_size(closure::ScalarDiffusivity) = 1 
 
function with_tracers(tracers, closure::ScalarDiffusivity{TD, Dir}) where {TD, Dir}
    κ = tracer_diffusivities(tracers, closure.κ)
    return ScalarDiffusivity{TD, Dir}(closure.ν, κ)
end

@inline diffusivity(closure::ScalarDiffusivity, ::Val{tracer_index}, args...) where tracer_index = closure.κ[tracer_index]
                        
Base.show(io::IO, closure::ScalarDiffusivity{TD, Dir})  where {TD, Dir}= 
    print(io, "ScalarDiffusivity:\n",
              "ν=$(closure.ν), κ=$(closure.κ)",
              "time discretization: $(time_discretization(closure))\n",
              "direction: $Dir")
