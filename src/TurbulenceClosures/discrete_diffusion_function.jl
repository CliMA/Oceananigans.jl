using Oceananigans.Operators: interpolate
using Oceananigans.Utils: instantiate

"""
    struct DiscreteDiffusionFunction{LX, LY, LZ, P, F} 

A wrapper for a diffusivity functions with optional parameters at a specified locations.

If LX == LY == LZ == nothing the function call requires locations in the signature
When `parameters=nothing`, the diffusivity `func` is called with the signature

```
func(i, j, k, grid, lx, ly, lz, clock, model_fields)
```

where `i, j, k` are the indices,
where `grid` is `model.grid`, `clock.time` is the current simulation time and
`clock.iteration` is the current model iteration, and
`model_fields` is a `NamedTuple` with `u, v, w`, the fields in `model.tracers` and the `model.auxiliary_fields`,

When `parameters` is not `nothing`, `func` is called with the signature

```
func(i, j, k, grid, lx, ly, lz, clock, model_fields, parameters)
```

If LX, LY, LZ != (nothing, nothing, nothing) the function call does not require locations in the signature
and the output will be automatically interpolated on the correct location

without parameters
```
func(i, j, k, grid, clock, model_fields)
```
with parameters
```
func(i, j, k, grid, clock, model_fields, parameters)
```
"""
struct DiscreteDiffusionFunction{LX, LY, LZ, P, F} 
    func :: F
    parameters :: P

    function DiscreteDiffusionFunction{LX, LY, LZ}(func::F, parameters::P) where {LX, LY, LZ, F, P}
        return new{LX, LY, LZ, P, F}(func, parameters)
    end
end

function DiscreteDiffusionFunction(func; parameters, loc)
    loc = instantiate.(loc)
    return DiscreteDiffusionFunction{typeof(loc[1]), typeof(loc[2]), typeof(loc[3])}(func, parameters)
end

const UnparameterizedDDF{LX, LY, LZ} = DiscreteDiffusionFunction{LX, LY, LZ, <:Nothing} where {LX, LY, LZ}
const UnlocalizedDDF                 = DiscreteDiffusionFunction{<:Nothing, <:Nothing, <:Nothing}
const UnlocalizedUnparametrizedDDF   = DiscreteDiffusionFunction{<:Nothing, <:Nothing, <:Nothing, <:Nothing}

@inline function getdiffusion(dd::DiscreteDiffusionFunction{LX, LY, LZ}, 
                              i, j, k, grid, location, clock, fields) where {LX, LY, LZ} 
        from = (LX(), LY(), LZ())
        return interpolate(i, j, k, grid, from, location, dd.func, clock, fields, dd.parameters)
end

@inline function getdiffusion(dd::UnparameterizedDDF{LX, LY, LZ}, 
                              i, j, k, grid, location, clock, fields) where {LX, LY, LZ} 
        from = (LX(), LY(), LZ())
    return interpolate(i, j, k, grid, from, location, dd.func, clock, fields)
end

@inline getdiffusion(dd::UnlocalizedDDF, i, j, k, grid, location, clock, fields) = 
        dd.func(i, j, k, grid, location..., clock, fields, dd.parametrs)

@inline getdiffusion(dd::UnlocalizedUnparametrizedDDF, i, j, k, grid, location, clock, fields) = 
        dd.func(i, j, k, grid, location..., clock, fields)

Adapt.adapt_structure(to, dd::DiscreteDiffusionFunction{LX, LY, LZ}) where {LX, LY, LZ} =
     DiscreteBoundaryFunction{LX, LY, LZ}(Adapt.adapt(to, dd.func),
                                          Adapt.adapt(to, dd.parameters))
