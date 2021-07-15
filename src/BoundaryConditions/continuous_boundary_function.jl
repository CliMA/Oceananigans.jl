using Oceananigans.Operators: index_and_interp_dependencies
using Oceananigans.Utils: tupleit, user_function_arguments

import Oceananigans: location

"""
    struct ContinuousBoundaryFunction{X, Y, Z, I, F, P, D, N, ℑ} <: Function

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
    function ContinuousBoundaryFunction(func::F, parameters::P, field_dependencies) where {F, P}
        field_dependencies = tupleit(field_dependencies)
        D = typeof(field_dependencies)
        return new{Nothing, Nothing, Nothing, Nothing, F, P, D, Nothing, Nothing}(func, parameters, field_dependencies, nothing, nothing)
    end

    function ContinuousBoundaryFunction{X, Y, Z, I}(func::F,
                                                    parameters::P,
                                                    field_dependencies::D,
                                                    field_dependencies_indices::N,
                                                    field_dependencies_interp::ℑ) where {X, Y, Z, I, F, P, D, ℑ, N}

        return new{X, Y, Z, I, F, P, D, N, ℑ}(func, parameters, field_dependencies, field_dependencies_indices, field_dependencies_interp)
    end
end

location(::ContinuousBoundaryFunction{X, Y, Z}) where {X, Y, Z} = X, Y, Z

#####
##### "Regularization" for IncompressibleModel setup
#####

"""
    regularize_boundary_condition(bc::BoundaryCondition{C, <:ContinuousBoundaryFunction},
                                  topo, loc, dim, I, prognostic_field_names) where C

Regularizes `bc.condition` for location `loc`, boundary index `I`, and `prognostic_field_names`,
returning `BoundaryCondition(C, regularized_condition)`.

The regularization of `bc.condition::ContinuousBoundaryFunction` requries

1. Setting the boundary location to `LX, LY, LZ`.
   The location in the boundary-normal direction is `Nothing`.

2. Setting the boundary-normal index `I` for indexing into `field_dependencies`.
   `I` is either `1` (for left boundaries) or
   `size(grid, n)` for a boundary in the `n`th direction where `n ∈ (1, 2, 3)` corresponds
   to `x, y, z`.

3. Determining the `indices` that map `model_fields` to `field_dependencies`.

4. Determining the `interps` functions that interpolate field_dependencies to the location
   of the boundary.
"""
function regularize_boundary_condition(bc::BoundaryCondition{C, <:ContinuousBoundaryFunction},
                                       topo, loc, dim, I, prognostic_field_names) where C

    boundary_func = bc.condition

    # Set boundary-normal location to Nothing:
    LX, LY, LZ = Tuple(i == dim ? Nothing : loc[i] for i = 1:3)

    indices, interps = index_and_interp_dependencies(LX, LY, LZ,
                                                     boundary_func.field_dependencies,
                                                     prognostic_field_names)

    regularized_boundary_func = ContinuousBoundaryFunction{LX, LY, LZ, I}(boundary_func.func,
                                                                          boundary_func.parameters,
                                                                          boundary_func.field_dependencies,
                                                                          indices, interps)

    return BoundaryCondition(C, regularized_boundary_func)
end

#####
##### Functionality for calling ContinuousBoundaryFunction in kernels
#####

const CBF = ContinuousBoundaryFunction

@inline boundary_index(i) = ifelse(i == 1, 1, i + 1) # convert near-boundary Center() index into boundary Face() index

@inline function _boundary_node_yz(i, j, k, grid, LY, LZ)
    i′ = boundary_index(i) # shift right index
    x, y, z = node(Face(), LY(), LZ(), i′, j, k, grid) # calculate node
    return y, z
end

@inline function _boundary_node_xz(i, j, k, grid, LY, LZ)
    j′ = boundary_index(j)
    x, y, z = node(LX(), Face(), LZ(), i, j′, k, grid)
    return x, z
end

@inline function _boundary_node_xy(i, j, k, grid, LX, LY)
    k′ = boundary_index(k)
    x, y, z = node(LX(), LY(), Face(), i, j, k′, grid)
    return x, y
end

@inline boundary_node_yz(i, j, k, grid, LY, LZ)        = _boundary_node_yz(i, j, k, grid, LY, LZ)
@inline boundary_node_yz(i, j, k, grid, ::Nothing, LZ) = tuple(_boundary_node_yz(i, j, k, grid, LY, LZ)[1])
@inline boundary_node_yz(i, j, k, grid, LY, ::Nothing) = tuple(_boundary_node_yz(i, j, k, grid, LY, LZ)[2])

@inline boundary_node_xz(i, j, k, grid, LX, LZ)        = _boundary_node_xz(i, j, k, grid, LX, LZ)
@inline boundary_node_xz(i, j, k, grid, ::Nothing, LZ) = tuple(_boundary_node_xz(i, j, k, grid, LX, LZ)[1])
@inline boundary_node_xz(i, j, k, grid, LX, ::Nothing) = tuple(_boundary_node_xz(i, j, k, grid, LX, LZ)[2])

@inline boundary_node_xy(i, j, k, grid, LX, LY)        = _boundary_node_xy(i, j, k, grid, LX, LY)
@inline boundary_node_xy(i, j, k, grid, ::Nothing, LY) = tuple(_boundary_node_xy(i, j, k, grid, LX, LY)[1])
@inline boundary_node_xy(i, j, k, grid, LY, ::Nothing) = tuple(_boundary_node_xy(i, j, k, grid, LX, LY)[2])

# Return ContinuousBoundaryFunction on east or west boundaries.
@inline function (bc::CBF{Nothing, LY, LZ, i})(j, k, grid, clock, model_fields) where {LY, LZ, i}
    args = user_function_arguments(i, j, k, grid, model_fields, bc.parameters, bc)
    yz = boundary_node_yz(i, j, k, grid, LY(), LZ())
    return bc.func(yz..., clock.time, args...)
end

# Return ContinuousBoundaryFunction on south or north boundaries.
@inline function (bc::CBF{LX, Nothing, LZ, j})(i, k, grid, clock, model_fields) where {LX, LZ, j}
    args = user_function_arguments(i, j, k, grid, model_fields, bc.parameters, bc)
    xz = boundary_node_xz(i, j, k, grid, LY(), LZ())
    return bc.func(xz..., clock.time, args...)
end

# Return ContinuousBoundaryFunction on bottom or top boundaries.
@inline function (bc::CBF{LX, LY, Nothing, k})(i, j, grid, clock, model_fields) where {LX, LY, k}
    args = user_function_arguments(i, j, k, grid, model_fields, bc.parameters, bc)
    xy = boundary_node_xy(i, j, k, grid, LY(), LZ())
    return bc.func(xy..., clock.time, args...)
end

# Signatures for ContinuousBounaryFunction called as auxiliary boundary functions with no clock or model_fields
@inline (bc::CBF{Nothing, LY, LZ, i, <:Any, <:Nothing})(j, k, grid) where {LY, LZ, i} = bc.func(boundary_node_yz(i, j, k, grid, LY(), LZ())...)
@inline (bc::CBF{LX, Nothing, LZ, j, <:Any, <:Nothing})(i, k, grid) where {LX, LZ, j} = bc.func(boundary_node_xz(i, j, k, grid, LX(), LZ())...)
@inline (bc::CBF{LX, LY, Nothing, k, <:Any, <:Nothing})(i, j, grid) where {LX, LY, k} = bc.func(boundary_node_xy(i, j, k, grid, LX(), LY())...)

@inline (bc::CBF{Nothing, LY, LZ, i})(j, k, grid) where {LY, LZ, i} = bc.func(boundary_node_yz(i, j, k, grid, LY(), LZ())..., bc.parameters)
@inline (bc::CBF{LX, Nothing, LZ, j})(i, k, grid) where {LX, LZ, j} = bc.func(boundary_node_xz(i, j, k, grid, LX(), LZ())..., bc.parameters)
@inline (bc::CBF{LX, LY, Nothing, k})(i, j, grid) where {LX, LY, k} = bc.func(boundary_node_xy(i, j, k, grid, LX(), LY())..., bc.parameters)

# Don't re-convert ContinuousBoundaryFunctions passed to BoundaryCondition constructor
BoundaryCondition(Classification::DataType, condition::ContinuousBoundaryFunction) = BoundaryCondition(Classification(), condition)
    
Adapt.adapt_structure(to, bf::ContinuousBoundaryFunction{LX, LY, LZ, I}) where {LX, LY, LZ, I} =
    ContinuousBoundaryFunction{LX, LY, LZ, I}(Adapt.adapt(to, bf.func),
                                              Adapt.adapt(to, bf.parameters),
                                              nothing,
                                              Adapt.adapt(to, bf.field_dependencies_indices),
                                              Adapt.adapt(to, bf.field_dependencies_interp))
