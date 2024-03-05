using Oceananigans.Operators: ℑxyz
using Oceananigans.Utils: instantiate

"""
    struct DiscreteDiffusionFunction{LX, LY, LZ, P, F} 

A wrapper for a diffusivity functions with optional parameters at a specified locations.
"""
struct DiscreteDiffusionFunction{LX, LY, LZ, P, F} 
    func :: F
    parameters :: P

    DiscreteDiffusionFunction{LX, LY, LZ}(func::F, parameters::P) where {LX, LY, LZ, F, P} =
        new{LX, LY, LZ, P, F}(func, parameters)
end

"""
    DiscreteDiffusionFunction(func; parameters, loc)

!!! info "Not user-facing method"
    This is not a user-facing method but instead it is used via various turbulence closures.
    Users build the diffusivities via each turbulence closure constructor, e.g., [`ScalarDiffusivity`](@ref).

Return a discrete representation of a diffusivity `func`tion with optional `parameters`
at specified `loc`ations.

Keyword Arguments
=================

* `parameters`: A named tuple with parameters used by `func`; default: `nothing`.

* `loc`: A tuple with the locations `(LX, LY, LZ)` that the diffusivity `func` is applied on.

  **Without locations**

  If `LX == LY == LZ == nothing` the diffusivity is evaluated at the required locations. In this case, the `func`tion
  call *requires passing* the locations `ℓx, ℓy, ℓz` in the signature:

  - Without parameters:

    ```julia
    func(i, j, k, grid, ℓx, ℓy, ℓz, clock, model_fields)
    ```

    where `i, j, k` are the indices, `grid` is `model.grid`, `ℓx, ℓy, ℓz` are the
    instantiated versions of `LX, LY, LZ`, `clock.time` is the current simulation time,
    `clock.iteration` is the current model iteration, and `model_fields` is a
    `NamedTuple` with `u, v, w`, the fields in `model.tracers` and the `model.auxiliary_fields`.

  - When `parameters` is not `nothing`, `func` is called with the signature

    ```julia
    func(i, j, k, grid, ℓx, ℓy, ℓz, clock, model_fields, parameters)
    ```

  **With locations**

  If `LX, LY, LZ != (nothing, nothing, nothing)` the diffusivity is evaluated at `(LX, LY, LZ)` and interpolated onto
  the required locations. In this case, the function call *does not require* locations in the signature. The diffusivity
  `func`ion is called with the signature:

  1. Without parameters:

    ```julia
    func(i, j, k, grid, clock, model_fields)
    ```

  2. When `parameters` is not `nothing`, `func` is called with the signature

    ```julia
    func(i, j, k, grid, clock, model_fields, parameters)
    ```
"""
function DiscreteDiffusionFunction(func; parameters, loc)
    loc = instantiate.(loc)
    return DiscreteDiffusionFunction{typeof(loc[1]), typeof(loc[2]), typeof(loc[3])}(func, parameters)
end

const UnparameterizedDDF{LX, LY, LZ} = DiscreteDiffusionFunction{LX, LY, LZ, <:Nothing} where {LX, LY, LZ}
const UnlocalizedDDF                 = DiscreteDiffusionFunction{<:Nothing, <:Nothing, <:Nothing}
const UnlocalizedUnparametrizedDDF   = DiscreteDiffusionFunction{<:Nothing, <:Nothing, <:Nothing, <:Nothing}

@inline function getdiffusivity(dd::DiscreteDiffusionFunction{LX, LY, LZ},
                              i, j, k, grid, location, clock, fields) where {LX, LY, LZ} 
    from = (LX(), LY(), LZ())
    return ℑxyz(i, j, k, grid, from, location, dd.func, clock, fields, dd.parameters)
end

@inline function getdiffusivity(dd::UnparameterizedDDF{LX, LY, LZ},
                              i, j, k, grid, location, clock, fields) where {LX, LY, LZ}
    from = (LX(), LY(), LZ())
    return ℑxyz(i, j, k, grid, from, location, dd.func, clock, fields)
end

@inline getdiffusivity(dd::UnlocalizedDDF, i, j, k, grid, location, clock, fields) =
        dd.func(i, j, k, grid, location..., clock, fields, dd.parameters)

@inline getdiffusivity(dd::UnlocalizedUnparametrizedDDF, i, j, k, grid, location, clock, fields) =
        dd.func(i, j, k, grid, location..., clock, fields)

Adapt.adapt_structure(to, dd::DiscreteDiffusionFunction{LX, LY, LZ}) where {LX, LY, LZ} =
     DiscreteBoundaryFunction{LX, LY, LZ}(Adapt.adapt(to, dd.func),
                                          Adapt.adapt(to, dd.parameters))
