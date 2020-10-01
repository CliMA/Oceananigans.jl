import Adapt

using Oceananigans: short_show
using Oceananigans.Operators: interpolation_operator
using Oceananigans.Fields: assumed_field_location, show_location
using Oceananigans.Utils: tupleit

"""
    ContinuousForcing{X, Y, Z, F, P, D, I}

A callable object that implements a "continuous form" forcing function
on a field at the location `X, Y, Z` with optional parameters.
"""
struct ContinuousForcing{X, Y, Z, F, P, D, I}
                    func :: F
              parameters :: P
      field_dependencies :: D
    ℑ_field_dependencies :: I

    function ContinuousForcing{X, Y, Z}(func, parameters, field_dependencies) where {X, Y, Z}

        field_dependencies = tupleit(field_dependencies)

        ℑ_field_dependencies = Tuple(interpolation_operator(assumed_field_location(name), (X, Y, Z))
                                     for name in field_dependencies)

        return new{X, Y, Z, 
                   typeof(func), 
                   typeof(parameters), 
                   typeof(field_dependencies),
                   typeof(ℑ_field_dependencies)}(func, parameters, field_dependencies, ℑ_field_dependencies)
                   
    end

    function ContinuousForcing{X, Y, Z}(func, parameters, field_dependencies, ℑ_field_dependencies) where {X, Y, Z}
        return new{X, Y, Z,
                   typeof(func),
                   typeof(parameters),
                   typeof(field_dependencies),
                   typeof(ℑ_field_dependencies)}(func, parameters, field_dependencies, ℑ_field_dependencies)
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
    ContinuousForcing{Cell, Cell, Cell}(func, parameters, field_dependencies)

@inline field_arguments(i, j, k, grid, model_fields, ℑ, field_names::NTuple{1}) =
    @inbounds (ℑ[1](i, j, k, grid, getproperty(model_fields, field_names[1])),)

@inline field_arguments(i, j, k, grid, model_fields, ℑ, field_names::NTuple{2}) =
    @inbounds (ℑ[1](i, j, k, grid, getproperty(model_fields, field_names[1])),
               ℑ[2](i, j, k, grid, getproperty(model_fields, field_names[2])))

@inline field_arguments(i, j, k, grid, model_fields, ℑ, field_names::NTuple{3}) =
    @inbounds (ℑ[1](i, j, k, grid, getproperty(model_fields, field_names[1])),
               ℑ[2](i, j, k, grid, getproperty(model_fields, field_names[2])),
               ℑ[3](i, j, k, grid, getproperty(model_fields, field_names[3])))

@inline field_arguments(i, j, k, grid, model_fields, ℑ, field_names::NTuple{4}) =
    @inbounds (ℑ[1](i, j, k, grid, getproperty(model_fields, field_names[1])),
               ℑ[2](i, j, k, grid, getproperty(model_fields, field_names[2])),
               ℑ[3](i, j, k, grid, getproperty(model_fields, field_names[3])),
               ℑ[4](i, j, k, grid, getproperty(model_fields, field_names[4])))

@inline field_arguments(i, j, k, grid, model_fields, ℑ, field_names::NTuple{N}) where N =
    ntuple(n -> ℑ[n](i, j, k, grid, getproperty(model_fields, field_names[n])), Val(N))

@inline function forcing_func_arguments(i, j, k, grid, model_fields, ::Nothing, forcing)

    ℑ = forcing.ℑ_field_dependencies
    dependencies = forcing.field_dependencies

    return field_arguments(i, j, k, grid, model_fields, ℑ, dependencies)
end

@inline function forcing_func_arguments(i, j, k, grid, model_fields, parameters, forcing)

    ℑ = forcing.ℑ_field_dependencies
    dependencies = forcing.field_dependencies
    parameters = forcing.parameters

    field_args = field_arguments(i, j, k, grid, model_fields, ℑ, dependencies)

    return tuple(field_args..., parameters)
end

@inline function (forcing::ContinuousForcing{X, Y, Z, F})(i, j, k, grid, clock, fields) where {X, Y, Z, F}

    args = forcing_func_arguments(i, j, k, grid, fields, forcing.parameters, forcing)

    return @inbounds forcing.func(xnode(X, i, grid), ynode(Y, j, grid), znode(Z, k, grid), clock.time, args...)
end

"""Show the innards of a `ContinuousForcing` in the REPL."""
Base.show(io::IO, forcing::ContinuousForcing{X, Y, Z, P}) where {X, Y, Z, P} =
    print(io, "ContinuousForcing{$P} at ", show_location(X, Y, Z), '\n',
        "├── func: $(short_show(forcing.func))", '\n',
        "├── parameters: $(forcing.parameters)", '\n',
        "└── field dependencies: $(forcing.field_dependencies)")

Adapt.adapt_structure(to, forcing::ContinuousForcing{X, Y, Z}) where {X, Y, Z} =
    ContinuousForcing{X, Y, Z}(Adapt.adapt(to, forcing.func),
                               Adapt.adapt(to, forcing.parameters),
                               Adapt.adapt(to, forcing.field_dependencies),
                               Adapt.adapt(to, forcing.ℑ_field_dependencies))
