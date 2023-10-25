using Oceananigans.Operators: index_and_interp_dependencies
using Oceananigans.Utils: tupleit, user_function_arguments
using Oceananigans.Grids: XFlatGrid, YFlatGrid, ZFlatGrid, YZFlatGrid, XZFlatGrid, XYFlatGrid
using Oceananigans.Grids: ξnode, ηnode, rnode

import Oceananigans: location
import Oceananigans.Utils: prettysummary

struct LeftBoundary end
struct RightBoundary end

"""
    struct ContinuousBoundaryFunction{X, Y, Z, I, F, P, D, N, ℑ} <: Function

A wrapper for the user-defined boundary condition function `func` at location
`X, Y, Z`. `I` denotes the boundary-normal index (`I=1` at western boundaries,
`I=grid.Nx` at eastern boundaries, etc). `F, P, D, N, ℑ` are, respectively, the 
user-defined function, parameters, field dependencies, indices of the field dependencies
in `model_fields`, and interpolation operators for interpolating `model_fields` to the
location at which the boundary condition is applied.
"""
struct ContinuousBoundaryFunction{X, Y, Z, S, F, P, D, N, ℑ}
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

    function ContinuousBoundaryFunction{X, Y, Z, S}(func::F,
                                                    parameters::P,
                                                    field_dependencies::D,
                                                    field_dependencies_indices::N,
                                                    field_dependencies_interp::ℑ) where {X, Y, Z, S, F, P, D, ℑ, N}

        return new{X, Y, Z, S, F, P, D, N, ℑ}(func, parameters, field_dependencies, field_dependencies_indices, field_dependencies_interp)
    end
end

location(::ContinuousBoundaryFunction{X, Y, Z}) where {X, Y, Z} = X, Y, Z

#####
##### "Regularization" for NonhydrostaticModel setup
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
                                       grid, loc, dim, Side, prognostic_field_names) where C

    boundary_func = bc.condition

    # Set boundary-normal location to Nothing:
    LX, LY, LZ = Tuple(i == dim ? Nothing : loc[i] for i = 1:3)

    indices, interps = index_and_interp_dependencies(LX, LY, LZ,
                                                     boundary_func.field_dependencies,
                                                     prognostic_field_names)

    regularized_boundary_func = ContinuousBoundaryFunction{LX, LY, LZ, Side}(boundary_func.func,
                                                                             boundary_func.parameters,
                                                                             boundary_func.field_dependencies,
                                                                             indices, interps)

    return BoundaryCondition(C, regularized_boundary_func)
end

@inline domain_boundary_indices(::LeftBoundary, N) = 1, 1
@inline domain_boundary_indices(::RightBoundary, N) = N, N + 1

@inline cell_boundary_index(::LeftBoundary, i) = i
@inline cell_boundary_index(::RightBoundary, i) = i + 1

#####
##### Kernel functions for "primitive grid" boundary conditions
#####

@inline x_boundary_node(i, j, k, grid,             ℓy, ℓz) = (ηnode(i, j, k, grid, Face(), ℓy, ℓz), rnode(i, j, k, grid, Face(), ℓy, ℓz))
@inline x_boundary_node(i, j, k, grid::YFlatGrid,  ℓy, ℓz) = tuple(rnode(i, j, k, grid, Face(), nothing, ℓz))
@inline x_boundary_node(i, j, k, grid::ZFlatGrid,  ℓy, ℓz) = tuple(ηnode(i, j, k, grid, Face(), ℓy, nothing))
@inline x_boundary_node(i, j, k, grid::YZFlatGrid, ℓy, ℓz) = tuple()

@inline y_boundary_node(i, j, k, grid,             ℓx, ℓz) = (ξnode(i, j, k, grid, ℓx, Face(), ℓz), rnode(i, j, k, grid, ℓx, Face(), ℓz))
@inline y_boundary_node(i, j, k, grid::XFlatGrid,  ℓx, ℓz) = tuple(rnode(i, j, k, grid, nothing, Face(), ℓz))
@inline y_boundary_node(i, j, k, grid::ZFlatGrid,  ℓx, ℓz) = tuple(ξnode(i, j, k, grid, ℓx, Face(), nothing))
@inline y_boundary_node(i, j, k, grid::XZFlatGrid, ℓx, ℓz) = tuple()

@inline z_boundary_node(i, j, k, grid,             ℓx, ℓy) = (ξnode(i, j, k, grid, ℓx, ℓy, Face()), ηnode(i, j, k, grid, ℓx, ℓy, Face()))
@inline z_boundary_node(i, j, k, grid::XFlatGrid,  ℓx, ℓy) = tuple(ηnode(i, j, k, grid, nothing, ℓy, Face()))
@inline z_boundary_node(i, j, k, grid::YFlatGrid,  ℓx, ℓy) = tuple(ξnode(i, j, k, grid, ℓx, nothing, Face()))
@inline z_boundary_node(i, j, k, grid::XYFlatGrid, ℓx, ℓy) = tuple()

const XBoundaryFunction{LY, LZ, S} = BoundaryCondition{<:Any, <:ContinuousBoundaryFunction{Nothing, LY, LZ, S}} where {LY, LZ, S}
const YBoundaryFunction{LX, LZ, S} = BoundaryCondition{<:Any, <:ContinuousBoundaryFunction{LX, Nothing, LZ, S}} where {LX, LZ, S}
const ZBoundaryFunction{LX, LY, S} = BoundaryCondition{<:Any, <:ContinuousBoundaryFunction{LX, LY, Nothing, S}} where {LX, LY, S}

# Return ContinuousBoundaryFunction on east or west boundaries.
@inline function getbc(bc::XBoundaryFunction{LY, LZ, S}, j::Integer, k::Integer,
                       grid::AbstractGrid, clock, model_fields, args...) where {LY, LZ, S}

    cbf = bc.condition
    i, i′ = domain_boundary_indices(S(), grid.Nx)
    args = user_function_arguments(i, j, k, grid, model_fields, cbf.parameters, cbf)
    X = x_boundary_node(i′, j, k, grid, LY(), LZ())

    return cbf.func(X..., clock.time, args...)
end

# Return ContinuousBoundaryFunction on south or north boundaries.
@inline function getbc(bc::YBoundaryFunction{LX, LZ, S}, i::Integer, k::Integer,
                       grid::AbstractGrid, clock, model_fields, args...) where {LX, LZ, S}

    cbf = bc.condition
    j, j′ = domain_boundary_indices(S(), grid.Ny)
    args = user_function_arguments(i, j, k, grid, model_fields, cbf.parameters, cbf)
    X = y_boundary_node(i, j′, k, grid, LX(), LZ())

    return cbf.func(X..., clock.time, args...)
end

# Return ContinuousBoundaryFunction on bottom or top boundaries.
@inline function getbc(bc::ZBoundaryFunction{LX, LY, S}, i::Integer, j::Integer,
                       grid::AbstractGrid, clock, model_fields, args...) where {LX, LY, S}

    cbf = bc.condition
    k, k′ = domain_boundary_indices(S(), grid.Nz)
    args = user_function_arguments(i, j, k, grid, model_fields, cbf.parameters, cbf)
    X = z_boundary_node(i, j, k′, grid, LX(), LY())

    return cbf.func(X..., clock.time, args...)
end

#####
##### For immersed boundary conditions
#####

# Return ContinuousBoundaryFunction on the east or west interface of a cell adjacent to an immersed boundary
@inline function getbc(bc::XBoundaryFunction{LY, LZ, S}, i::Integer, j::Integer, k::Integer,
                       grid::AbstractGrid, clock, model_fields, args...) where {LY, LZ, S}

    cbf = bc.condition
    i′ = cell_boundary_index(S(), i)
    args = user_function_arguments(i, j, k, grid, model_fields, cbf.parameters, cbf)
    X = node(i′, j, k, grid, Face(), LY(), LZ())

    return cbf.func(X..., clock.time, args...)
end

# Return ContinuousBoundaryFunction on the south or north interface of a cell adjacent to an immersed boundary
@inline function getbc(bc::YBoundaryFunction{LX, LZ, S}, i::Integer, j::Integer, k::Integer,
                       grid::AbstractGrid, clock, model_fields, args...) where {LX, LZ, S}

    cbf = bc.condition
    j′ = cell_boundary_index(S(), j)
    args = user_function_arguments(i, j, k, grid, model_fields, cbf.parameters, cbf)
    X = node(i, j′, k, grid, LX(), Face(), LZ())

    return cbf.func(X..., clock.time, args...)
end

# Return ContinuousBoundaryFunction on the bottom or top interface of a cell adjacent to an immersed boundary
@inline function getbc(bc::ZBoundaryFunction{LX, LY, S}, i::Integer, j::Integer, k::Integer,
                       grid::AbstractGrid, clock, model_fields, args...) where {LX, LY, S}

    cbf = bc.condition
    k′ = cell_boundary_index(S(), k)
    args = user_function_arguments(i, j, k, grid, model_fields, cbf.parameters, cbf)
    X = node(i, j, k′, grid, LX(), LY(), Face())

    return cbf.func(X..., clock.time, args...)
end

#####
##### Utils
#####

# Don't re-convert ContinuousBoundaryFunctions passed to BoundaryCondition constructor
BoundaryCondition(Classification::DataType, condition::ContinuousBoundaryFunction) = BoundaryCondition(Classification(), condition)

# TODO: show parameter, field dependencies, etc
function Base.summary(bf::ContinuousBoundaryFunction)
    loc = location(bf)
    return string("ContinuousBoundaryFunction ", prettysummary(bf.func, false), " at ", loc)
end

prettysummary(bf::ContinuousBoundaryFunction) = summary(bf)
    
Adapt.adapt_structure(to, bf::ContinuousBoundaryFunction{LX, LY, LZ, S}) where {LX, LY, LZ, S} =
    ContinuousBoundaryFunction{LX, LY, LZ, S}(Adapt.adapt(to, bf.func),
                                              Adapt.adapt(to, bf.parameters),
                                              nothing,
                                              Adapt.adapt(to, bf.field_dependencies_indices),
                                              Adapt.adapt(to, bf.field_dependencies_interp))
