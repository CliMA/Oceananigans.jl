using Oceananigans.Operators: index_and_interp_dependencies, assumed_field_location

"""
    ContinuousBoundaryFunction{X, Y, Z, I, F, P, D, N, ℑ} <: Function

A wrapper for the user-defined boundary condition function `func` at location
`X, Y, Z`. `I` denotes the boundary-normal index (`I=1` at western boundaries,
`I=grid.Nx` at eastern boundaries, etc). `F, P, D, N, ℑ` are, respectively, the 
user-defined function, parameters, field dependencies, indices of the field dependencies
in `model_fields`, and interpolation operators for interpolating `model_fields` to the
location at which the boundary condition is applied.
"""
struct ContinuousBoundaryFunction{X, Y, Z, I, F, P, D, N, ℑ} <: Function
                          func :: F
                    parameters :: P
            field_dependencies :: D
    field_dependencies_indices :: N
     field_dependencies_interp :: ℑ

    """ Returns a location-less wrapper for `func`, `parameters`, and `field_dependencies`."""
    function ContinuousBoundaryFunction(func, parameters, field_dependencies)
        return new{Nothing, Nothing, Nothing, Nothing,
                   typeof(func), typeof(parameters), Nothing, Nothing, Nothing}(func, parameters, field_dependencies, nothing, nothing)
    end

    """ Returns a wrapper with location `X, Y, Z` for `func`, `parameters`, and `field_dependencies`."""
    function ContinuousBoundaryFunction{X, Y, Z}(func, parameters, field_dependencies) where {X, Y, Z}
        return new{X, Y, Z, Nothing,
                   typeof(func),
                   typeof(parameters),
                   typeof(field_dependencies),
                   Nothing, Nothing}(func, parameters, field_dependencies, nothing, nothing)
    end

    function ContinuousBoundaryFunction{X, Y, Z, I}(func, parameters, field_dependencies,
                                                    field_dependencies_indices, field_dependencies_interp) where {X, Y, Z, I}
        return new{X, Y, Z, I,
                   typeof(func),
                   typeof(parameters),
                   typeof(field_dependencies),
                   typeof(field_dependencies_indices),
                   typeof(field_dependencies_interp)}(func,
                                                      parameters,
                                                      field_dependencies,
                                                      field_dependencies_indices,
                                                      field_dependencies_interp)
    end
end

regularize_boundary_condition(bc, I, field_name, model_field_names) = bc # fallback

"""
    regularize_boundary_condition(bc::BoundaryCondition{C, <:ContinuousBoundaryFunction},
                                  I, field_name, model_field_names) where C

Regularizes `bc.condition` for boundary index `I` by determining the indices of
`bc.condition.field_dependencies` in `model_field_names` and associated interpolation functions
so that `bc` can be used during time-stepping `IncompressibleModel`.
"""
function regularize_boundary_condition(bc::BoundaryCondition{C, <:ContinuousBoundaryFunction}, I, field_name, model_field_names) where C
    boundary_func = bc.condition

    X, Y, Z = assumed_field_location(field_name)

    indices, interps = index_and_interp_dependencies(X, Y, Z,
                                                     boundary_func.field_dependencies,
                                                     model_field_names)

    regularized_boundary_func = ContinuousBoundaryFunction{X, Y, Z, I}(boundary_func.func,
                                                                       boundary_func.parameters,
                                                                       boundary_func.field_dependencies,
                                                                       indices, interps)

    return BoundaryCondition(C, regularized_boundary_func)
end

function assign_location(bc::BoundaryCondition{C, <:ContinuousBoundaryFunction}, X, Y, Z) where C
    condition = ContinuousBoundaryFunction{X, Y, Z}(bc.condition.func,
                                                    bc.condition.parameters,
                                                    bc.condition.field_dependencies)
    return BoundaryCondition(C, condition)
end

@inline function (bc::ContinuousBoundaryFunction{Nothing, Y, Z, i})(j, k, grid, clock, model_fields) where {Y, Z, i}
    args = user_function_arguments(i, j, k, grid, model_fields, bc.parameters, bc)
    return bc.func(ynode(Y, j, grid), znode(Z, k, grid), clock.time, args...)
end

@inline function (bc::ContinuousBoundaryFunction{X, Nothing, Z, j})(i, k, grid, clock, model_fields) where {X, Z, j}
    args = user_function_arguments(i, j, k, grid, model_fields, bc.parameters, bc)
    return bc.func(xnode(X, i, grid), znode(Z, k, grid), clock.time, args...)
end

@inline function (bc::ContinuousBoundaryFunction{X, Y, Nothing, k})(i, j, grid, clock, model_fields) where {X, Y, k}
    args = user_function_arguments(i, j, k, grid, model_fields, bc.parameters, bc)
    return bc.func(xnode(X, i, grid), ynode(Y, j, grid), clock.time, args...)
end

# Don't re-convert ContinuousBoundaryFunctions passed to BoundaryCondition constructor
BoundaryCondition(TBC, condition::ContinuousBoundaryFunction) =
    BoundaryCondition{TBC, typeof(condition)}(condition)
