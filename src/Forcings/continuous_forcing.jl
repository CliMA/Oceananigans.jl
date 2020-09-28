using Oceananigans.Operators: interpolation_operator
using Oceananigans.Fields: assumed_field_location, show_location
using Oceananigans.Utils: tupleit

"""
    ContinuousForcing{X, Y, Z, P, N, F, I}

A callable object that implements a "continuous form" forcing function
on a field at the location `X, Y, Z` with optional parameters.
"""
struct ContinuousForcing{X, Y, Z, P, N, F, I}
                    func :: F
              parameters :: P
      field_dependencies :: NTuple{N, Symbol}
    ℑ_field_dependencies :: I

    function ContinuousForcing{X, Y, Z}(func, parameters, field_dependencies) where {X, Y, Z}

        field_dependencies = tupleit(field_dependencies)

        ℑ_field_dependencies = Tuple(interpolation_operator(assumed_field_location(name), (X, Y, Z))
                                     for name in field_dependencies)

        return new{X, Y, Z, typeof(parameters), length(field_dependencies),
                   typeof(func), typeof(ℑ_field_dependencies)}(func, parameters, field_dependencies, ℑ_field_dependencies)
                   
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

@inline function field_arguments(i, j, k, grid, fields, ℑfields, field_names::NTuple{N, Symbol}) where N

    return ntuple(n -> ℑfields[n](i, j, k, grid, getproperty(fields, field_names[n])), Val(N))
end

@inline forcing_func_arguments(i, j, k, grid, fields, forcing::ContinuousForcing{X, Y, Z, <:Nothing}) where {X, Y, Z} =
    field_arguments(i, j, k, grid,
                    fields,
                    forcing.ℑ_field_dependencies,
                    forcing.field_dependencies)

@inline function forcing_func_arguments(i, j, k, grid, fields, forcing::ContinuousForcing{X, Y, Z}) where {X, Y, Z}

    field_args = field_arguments(i, j, k, grid,
                                 fields,
                                 forcing.ℑ_field_dependencies,
                                 forcing.field_dependencies)

    return tuple(field_args..., forcing.parameters)
end

@inline (forcing::ContinuousForcing{X, Y, Z})(i, j, k, grid, clock, fields) where {X, Y, Z} =
    @inbounds forcing.func(xnode(X, i, grid),
                           ynode(Y, j, grid),
                           znode(Z, k, grid),
                           clock.time,
                           forcing_func_arguments(i, j, k, grid, fields, forcing)...)

"""Show the innards of a `ContinuousForcing` in the REPL."""
Base.show(io::IO, forcing::ContinuousForcing{X, Y, Z, P}) where {X, Y, Z, P} =
    print(io, "ContinuousForcing{$P} at ", show_location(X, Y, Z), '\n',
        "├── func: $(string(Symbol(forcing.func)))", '\n',
        "├── parameters: $(forcing.parameters)", '\n',
        "└── field dependencies: $(forcing.field_dependencies)")
