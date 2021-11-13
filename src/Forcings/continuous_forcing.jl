import Adapt

using Oceananigans: short_show
using Oceananigans.Grids: node
using Oceananigans.Operators: assumed_field_location, index_and_interp_dependencies
using Oceananigans.Fields: show_location
using Oceananigans.Utils: user_function_arguments, tupleit

"""
    ContinuousForcing{LX, LY, LZ, P, F, D, I, ℑ}

A callable object that implements a "continuous form" forcing function
on a field at the location `LX, LY, LZ` with optional parameters.
"""
struct ContinuousForcing{LX, LY, LZ, P, F, D, I, ℑ}
                          func :: F
                    parameters :: P
            field_dependencies :: D
    field_dependencies_indices :: I
     field_dependencies_interp :: ℑ

    # Non-public "temporary" constructor that stores func, parameters, and field_dependencies
    # for later regularization
    function ContinuousForcing(func, parameters, field_dependencies)
        field_dependencies = tupleit(field_dependencies)

        return new{Nothing, Nothing, Nothing,
                   typeof(parameters),
                   typeof(func),
                   typeof(field_dependencies),
                   Nothing,
                   Nothing}(func, parameters, field_dependencies, nothing, nothing)
    end

    # Non-public "final" constructor.
    function ContinuousForcing{LX, LY, LZ}(func, parameters=nothing, field_dependencies=(),
                                        field_dependencies_indices=(), field_dependencies_interp=()) where {LX, LY, LZ}
        return new{LX, LY, LZ,
                   typeof(parameters),
                   typeof(func),
                   typeof(field_dependencies),
                   typeof(field_dependencies_indices),
                   typeof(field_dependencies_interp)}(func, parameters, field_dependencies,
                                                      field_dependencies_indices, field_dependencies_interp)
    end
end

"""
    ContinuousForcing(func; parameters=nothing, field_dependencies=())

Construct a "continuous form" forcing with optional `parameters` and optional
`field_dependencies` on other fields in a model.

If neither `parameters` nor `field_dependencies` are provided, then `func` must be
callable with the signature

    `func(x, y, z, t)`

where `x, y, z` are the east-west, north-south, and vertical spatial coordinates, and `t` is time.

If `field_dependencies` are provided, the signature of `func` must include them.
For example, if `field_dependencies=(:u, :S)` (and `parameters` are _not_ provided), then
`func` must be callable with the signature

    `func(x, y, z, t, u, S)`

where `u` is assumed to be the `u`-velocity component, and `S` is a tracer. Note that any field
which does not have the name `u`, `v`, or `w` is assumed to be a tracer and must be present
in `model.tracers`.

If `parameters` are provided, then the _last_ argument to `func` must be `parameters`.
For example, if `func` has no `field_dependencies` but does depend on `parameters`, then
it must be callable with the signature

    `func(x, y, z, t, parameters)`

With `field_dependencies=(:u, :v, :w, :c)` and `parameters`, then `func` must be
callable with the signature

    `func(x, y, z, t, u, v, w, c, parameters)`

"""
ContinuousForcing(func; parameters=nothing, field_dependencies=()) =
    ContinuousForcing(func, parameters, field_dependencies)

"""
    regularize_forcing(forcing::ContinuousForcing, field, field_name, model_field_names)

Regularize `forcing::ContinuousForcing` by determining the indices of `forcing.field_dependencies`
in `model_field_names`, and associated interpolation functions so `forcing` can be used during
time-stepping `NonhydrostaticModel`.
"""
function regularize_forcing(forcing::ContinuousForcing, field, field_name, model_field_names)

    LX, LY, LZ = location(field)

    indices, interps = index_and_interp_dependencies(LX, LY, LZ,
                                                     forcing.field_dependencies,
                                                     model_field_names)

    return ContinuousForcing{LX, LY, LZ}(forcing.func, forcing.parameters,
                                         forcing.field_dependencies, indices, interps)
end

#####
##### Functions for calling ContinuousForcing in a time-stepping kernel
#####

@inline function (forcing::ContinuousForcing{LX, LY, LZ, P, F})(i, j, k, grid, clock, model_fields) where {LX, LY, LZ, P, F}

    args = user_function_arguments(i, j, k, grid, model_fields, forcing.parameters, forcing)

    x = xnode(LX(), LY(), LZ(), i, j, k, grid)
    y = ynode(LX(), LY(), LZ(), i, j, k, grid)
    z = znode(LX(), LY(), LZ(), i, j, k, grid)

    return forcing.func(x, y, z, clock.time, args...)
end

"""Show the innards of a `ContinuousForcing` in the REPL."""
Base.show(io::IO, forcing::ContinuousForcing{LX, LY, LZ, P}) where {LX, LY, LZ, P} =
    print(io, "ContinuousForcing{$P} at ", show_location(LX, LY, LZ), '\n',
        "├── func: $(short_show(forcing.func))", '\n',
        "├── parameters: $(forcing.parameters)", '\n',
        "└── field dependencies: $(forcing.field_dependencies)")

"""Show the innards of an "non-regularized" `ContinuousForcing` in the REPL."""
Base.show(io::IO, forcing::ContinuousForcing{Nothing, Nothing, Nothing, P}) where P =
    print(io, "ContinuousForcing{$P}", '\n',
        "├── func: $(short_show(forcing.func))", '\n',
        "├── parameters: $(forcing.parameters)", '\n',
        "└── field dependencies: $(forcing.field_dependencies)")

Adapt.adapt_structure(to, forcing::ContinuousForcing{LX, LY, LZ}) where {LX, LY, LZ} =
    ContinuousForcing{LX, LY, LZ}(Adapt.adapt(to, forcing.func),
                               Adapt.adapt(to, forcing.parameters),
                               nothing,
                               Adapt.adapt(to, forcing.field_dependencies_indices),
                               Adapt.adapt(to, forcing.field_dependencies_interp))

