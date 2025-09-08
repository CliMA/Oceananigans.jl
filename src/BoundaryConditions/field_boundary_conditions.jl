using Oceananigans.Operators: assumed_field_location
using Oceananigans.Grids: YFlatGrid
using GPUArraysCore

#####
##### Default boundary conditions
#####

struct DefaultBoundaryCondition{BC}
    boundary_condition :: BC
end

DefaultBoundaryCondition() = DefaultBoundaryCondition(NoFluxBoundaryCondition())

default_prognostic_bc(::Grids.Periodic, loc,      default)  = PeriodicBoundaryCondition()
default_prognostic_bc(::FullyConnected, loc,      default)  = MultiRegionCommunicationBoundaryCondition()
default_prognostic_bc(::Flat,           loc,      default)  = nothing
default_prognostic_bc(::Bounded,        ::Center, default)  = default.boundary_condition
default_prognostic_bc(::LeftConnected,  ::Center, default)  = default.boundary_condition
default_prognostic_bc(::RightConnected, ::Center, default)  = default.boundary_condition

# TODO: make model constructors enforce impenetrability on velocity components to simplify this code
default_prognostic_bc(::Bounded,        ::Face, default) = ImpenetrableBoundaryCondition()
default_prognostic_bc(::LeftConnected,  ::Face, default) = ImpenetrableBoundaryCondition()
default_prognostic_bc(::RightConnected, ::Face, default) = ImpenetrableBoundaryCondition()

default_prognostic_bc(::Bounded,        ::Nothing, default) = nothing
default_prognostic_bc(::Flat,           ::Nothing, default) = nothing
default_prognostic_bc(::Grids.Periodic, ::Nothing, default) = nothing
default_prognostic_bc(::FullyConnected, ::Nothing, default) = nothing
default_prognostic_bc(::LeftConnected,  ::Nothing, default) = nothing
default_prognostic_bc(::RightConnected, ::Nothing, default) = nothing

_default_auxiliary_bc(topo, loc) = default_prognostic_bc(topo, loc, DefaultBoundaryCondition())
_default_auxiliary_bc(::Bounded, ::Face)        = nothing
_default_auxiliary_bc(::RightConnected, ::Face) = nothing
_default_auxiliary_bc(::LeftConnected,  ::Face) = nothing

default_auxiliary_bc(grid, ::Val{:east}, loc)   = _default_auxiliary_bc(topology(grid, 1)(), loc[1])
default_auxiliary_bc(grid, ::Val{:west}, loc)   = _default_auxiliary_bc(topology(grid, 1)(), loc[1])
default_auxiliary_bc(grid, ::Val{:south}, loc)  = _default_auxiliary_bc(topology(grid, 2)(), loc[2])
default_auxiliary_bc(grid, ::Val{:north}, loc)  = _default_auxiliary_bc(topology(grid, 2)(), loc[2])
default_auxiliary_bc(grid, ::Val{:bottom}, loc) = _default_auxiliary_bc(topology(grid, 3)(), loc[3])
default_auxiliary_bc(grid, ::Val{:top}, loc)    = _default_auxiliary_bc(topology(grid, 3)(), loc[3])

#####
##### Field boundary conditions
#####

mutable struct FieldBoundaryConditions{W, E, S, N, B, T, I, K, O}
    west :: W
    east :: E
    south :: S
    north :: N
    bottom :: B
    top :: T
    immersed :: I
    kernels :: K # kernels used to fill halo regions
    ordered_bcs :: O
end

const boundarynames = (:west, :east, :south, :north, :bottom, :top, :immersed)

const NoKernelFBC = FieldBoundaryConditions{W, E, S, N, B, T, I, Nothing, Nothing} where {W, E, S, N, B, T, I}

# Internal constructor that fills up computational details in the "auxiliaries" spot.
function FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed, nothing, nothing)
end

function FieldBoundaryConditions(indices::Tuple, west, east, south, north, bottom, top, immersed)
    # Turn bcs in windowed dimensions into nothing
    west, east   = window_boundary_conditions(indices[1], west, east)
    south, north = window_boundary_conditions(indices[2], south, north)
    bottom, top  = window_boundary_conditions(indices[3], bottom, top)
    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

FieldBoundaryConditions(indices::Tuple, bcs::FieldBoundaryConditions) =
    FieldBoundaryConditions(indices, (getproperty(bcs, side) for side in boundarynames)...)

FieldBoundaryConditions(indices::Tuple, ::Nothing) = nothing
FieldBoundaryConditions(indices::Tuple, ::Missing) = nothing

# return boundary conditions only if the field is not windowed!
window_boundary_conditions(::UnitRange,  left, right) = nothing, nothing
window_boundary_conditions(::Base.OneTo, left, right) = nothing, nothing
window_boundary_conditions(::Colon,      left, right) = left, right

# The only thing we need
Adapt.adapt_structure(to, fbcs::FieldBoundaryConditions) = (kernels = fbcs.kernels, ordered_bcs = Adapt.adapt(to, fbcs.ordered_bcs))

on_architecture(arch, fbcs::FieldBoundaryConditions) =
    FieldBoundaryConditions(on_architecture(arch, fbcs.west),
                            on_architecture(arch, fbcs.east),
                            on_architecture(arch, fbcs.south),
                            on_architecture(arch, fbcs.north),
                            on_architecture(arch, fbcs.bottom),
                            on_architecture(arch, fbcs.top),
                            on_architecture(arch, fbcs.immersed), 
                            fbcs.kernels,
                            on_architecture(arch, fbcs.ordered_bcs))

"""
    FieldBoundaryConditions(; kwargs...)

Return a template for boundary conditions on prognostic fields.

Keyword arguments
=================

Keyword arguments specify boundary conditions on the 7 possible boundaries:

- `west`: left end point in the `x`-direction where `i = 1`
- `east`: right end point in the `x`-direction where `i = grid.Nx`
- `south`: left end point in the `y`-direction where `j = 1`
- `north`: right end point in the `y`-direction where `j = grid.Ny`
- `bottom`: right end point in the `z`-direction where `k = 1`
- `top`: right end point in the `z`-direction where `k = grid.Nz`
- `immersed`: boundary between solid and fluid for immersed boundaries

If a boundary condition is unspecified, the default for prognostic fields
and the topology in the boundary-normal direction is used:

 - `PeriodicBoundaryCondition` for `Periodic` directions
 - `NoFluxBoundaryCondition` for `Bounded` directions and `Centered`-located fields
 - `ImpenetrableBoundaryCondition` for `Bounded` directions and `Face`-located fields
 - `nothing` for `Flat` directions and/or `Nothing`-located fields
"""
FieldBoundaryConditions(default_bounded_bc::BoundaryCondition = NoFluxBoundaryCondition();
                        west = DefaultBoundaryCondition(default_bounded_bc),
                        east = DefaultBoundaryCondition(default_bounded_bc),
                        south = DefaultBoundaryCondition(default_bounded_bc),
                        north = DefaultBoundaryCondition(default_bounded_bc),
                        bottom = DefaultBoundaryCondition(default_bounded_bc),
                        top = DefaultBoundaryCondition(default_bounded_bc),
                        immersed = DefaultBoundaryCondition(default_bounded_bc)) =
    FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)

"""
    FieldBoundaryConditions(grid, location, indices=(:, :, :);
                            west     = default_auxiliary_bc(grid, boundary, loc),
                            east     = default_auxiliary_bc(grid, boundary, loc),
                            south    = default_auxiliary_bc(grid, boundary, loc),
                            north    = default_auxiliary_bc(grid, boundary, loc),
                            bottom   = default_auxiliary_bc(grid, boundary, loc),
                            top      = default_auxiliary_bc(grid, boundary, loc),
                            immersed = NoFluxBoundaryCondition())

Return boundary conditions for auxiliary fields (fields whose values are
derived from a model's prognostic fields) on `grid` and at `location`.

Keyword arguments
=================

Keyword arguments specify boundary conditions on the 6 possible boundaries:

- `west`, left end point in the `x`-direction where `i = 1`
- `east`, right end point in the `x`-direction where `i = grid.Nx`
- `south`, left end point in the `y`-direction where `j = 1`
- `north`, right end point in the `y`-direction where `j = grid.Ny`
- `bottom`, right end point in the `z`-direction where `k = 1`
- `top`, right end point in the `z`-direction where `k = grid.Nz`
- `immersed`: boundary between solid and fluid for immersed boundaries

If a boundary condition is unspecified, the default for auxiliary fields
and the topology in the boundary-normal direction is used:

- `PeriodicBoundaryCondition` for `Periodic` directions
- `GradientBoundaryCondition(0)` for `Bounded` directions and `Centered`-located fields
- `nothing` for `Bounded` directions and `Face`-located fields
- `nothing` for `Flat` directions and/or `Nothing`-located fields
"""
function FieldBoundaryConditions(grid::AbstractGrid, loc, indices=(:, :, :);
                                 west     = default_auxiliary_bc(grid, Val(:west),   loc),
                                 east     = default_auxiliary_bc(grid, Val(:east),   loc),
                                 south    = default_auxiliary_bc(grid, Val(:south),  loc),
                                 north    = default_auxiliary_bc(grid, Val(:north),  loc),
                                 bottom   = default_auxiliary_bc(grid, Val(:bottom), loc),
                                 top      = default_auxiliary_bc(grid, Val(:top),    loc),
                                 immersed = DefaultBoundaryCondition())

    bcs = FieldBoundaryConditions(indices, west, east, south, north, bottom, top, immersed)
    return regularize_field_boundary_conditions(bcs, grid, loc)
end

#####
##### Boundary condition "regularization"
#####
##### TODO: this probably belongs in Oceananigans.Models
#####

function regularize_immersed_boundary_condition(ibc, grid, loc, field_name, args...)
    if !(ibc isa DefaultBoundaryCondition || isnothing(ibc))
        msg = """$field_name was assigned an immersed boundary condition
              $ibc,
              but this is not supported on
              $(summary(grid)).
              The immersed boundary condition on $field_name will have no effect.
              """

        @warn msg
    end

    return nothing
end

  regularize_west_boundary_condition(bc, args...) = regularize_boundary_condition(bc, args...)
  regularize_east_boundary_condition(bc, args...) = regularize_boundary_condition(bc, args...)
 regularize_south_boundary_condition(bc, args...) = regularize_boundary_condition(bc, args...)
 regularize_north_boundary_condition(bc, args...) = regularize_boundary_condition(bc, args...)
regularize_bottom_boundary_condition(bc, args...) = regularize_boundary_condition(bc, args...)
   regularize_top_boundary_condition(bc, args...) = regularize_boundary_condition(bc, args...)

# regularize default boundary conditions
function regularize_boundary_condition(default::DefaultBoundaryCondition, grid, loc, dim, args...)
    default_bc = default_prognostic_bc(topology(grid, dim)(), loc[dim], default)
    return regularize_boundary_condition(default_bc, grid, loc, dim, args...)
end

regularize_boundary_condition(bc, args...) = bc # fallback

# Convert all `Number` boundary conditions to `eltype(grid)`
regularize_boundary_condition(bc::BoundaryCondition{C, <:Number}, grid, args...) where C =
    BoundaryCondition(bc.classification, convert(eltype(grid), bc.condition))

"""
    regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                         grid::AbstractGrid,
                                         field_name::Symbol,
                                         prognostic_names=nothing)

Compute default boundary conditions and attach field locations to ContinuousBoundaryFunction
boundary conditions for prognostic model field boundary conditions.

!!! warn "Immersed `ContinuousBoundaryFunction` is unsupported"
    `ContinuousBoundaryFunction` is not supported on immersed boundaries.
    We therefore do not regularize the immersed boundary condition.
"""
function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                              grid::AbstractGrid,
                                              field_name::Symbol,
                                              prognostic_names=nothing)

    loc = assumed_field_location(field_name)
    return regularize_field_boundary_conditions(bcs, grid, loc, prognostic_names, field_name)
end

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                              grid::AbstractGrid,
                                              loc::Tuple,
                                              prognostic_names=nothing,
                                              field_name=nothing)
    
    west   = regularize_west_boundary_condition(bcs.west,     grid, loc, 1, LeftBoundary,  prognostic_names)
    east   = regularize_east_boundary_condition(bcs.east,     grid, loc, 1, RightBoundary, prognostic_names)
    south  = regularize_south_boundary_condition(bcs.south,   grid, loc, 2, LeftBoundary,  prognostic_names)
    north  = regularize_north_boundary_condition(bcs.north,   grid, loc, 2, RightBoundary, prognostic_names)
    bottom = regularize_bottom_boundary_condition(bcs.bottom, grid, loc, 3, LeftBoundary,  prognostic_names)
    top    = regularize_top_boundary_condition(bcs.top,       grid, loc, 3, RightBoundary, prognostic_names)

    immersed = regularize_immersed_boundary_condition(bcs.immersed, grid, loc, field_name, prognostic_names)

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

# For nested NamedTuples of boundary conditions (eg diffusivity boundary conditions)
function regularize_field_boundary_conditions(boundary_conditions::NamedTuple,
                                              grid::AbstractGrid,
                                              group_name::Symbol,
                                              prognostic_names=nothing)

    return NamedTuple(field_name => regularize_field_boundary_conditions(field_bcs, grid, field_name, prognostic_names)
                      for (field_name, field_bcs) in pairs(boundary_conditions))
end

regularize_field_boundary_conditions(::Missing,
                                     grid::AbstractGrid,
                                     field_name::Symbol,
                                     prognostic_names=nothing) = missing

#####
##### Outer interface for model constructors
#####

regularize_field_boundary_conditions(boundary_conditions::NamedTuple, grid::AbstractGrid, prognostic_names::Tuple) =
    NamedTuple(field_name => regularize_field_boundary_conditions(field_bcs, grid, field_name, prognostic_names)
               for (field_name, field_bcs) in pairs(boundary_conditions))

#####
##### Special behavior for LatitudeLongitudeGrid
#####

# TODO: these may be incorrect because we have not defined behavior for prognostic fields (which are
# treated by `regularize`).
regularize_north_boundary_condition(bc::DefaultBoundaryCondition, grid::LatitudeLongitudeGrid, loc, args...) =
    regularize_boundary_condition(default_prognostic_bc(grid, Val(:north), loc, bc), grid, loc, args...)

regularize_south_boundary_condition(bc::DefaultBoundaryCondition, grid::LatitudeLongitudeGrid, loc, args...) =
    regularize_boundary_condition(default_prognostic_bc(grid, Val(:south), loc, bc), grid, loc, args...)

function default_prognostic_bc(grid::LatitudeLongitudeGrid, ::Val{:north}, (ℓx, ℓy, ℓz), default)
    φnorth = @allowscalar φnode(grid.Ny+1, grid, Face())
    default_bc = default_prognostic_bc(topology(grid, 2)(), ℓy, default)
    return φnorth ≈ 90 ? maybe_polar_boundary_condition(grid, :north, ℓy, ℓz) : default_bc
end

function default_prognostic_bc(grid::LatitudeLongitudeGrid, ::Val{:south}, (ℓx, ℓy, ℓz), default)
    φsouth = @allowscalar φnode(1, grid, Face())
    default_bc = default_prognostic_bc(topology(grid, 2)(), ℓy, default)
    return φsouth ≈ -90 ? maybe_polar_boundary_condition(grid, :south, ℓy, ℓz) : default_bc
end

function default_auxiliary_bc(grid::LatitudeLongitudeGrid, ::Val{:north}, (ℓx, ℓy, ℓz))
    φnorth = @allowscalar φnode(grid.Ny+1, grid, Face())
    default_bc = _default_auxiliary_bc(topology(grid, 2)(), ℓy)
    return φnorth ≈ 90 ? maybe_polar_boundary_condition(grid, :north, ℓy, ℓz) : default_bc
end

function default_auxiliary_bc(grid::LatitudeLongitudeGrid, ::Val{:south}, (ℓx, ℓy, ℓz))
    φsouth = @allowscalar φnode(1, grid, Face())
    default_bc = _default_auxiliary_bc(topology(grid, 2)(), ℓy)
    return φsouth ≈ -90 ? maybe_polar_boundary_condition(grid, :south, ℓy, ℓz) : default_bc
end

default_prognostic_bc(grid::LatitudeLongitudeGrid{<:Any, <:Any, Flat}, ::Val{:north}, loc, default) = default
default_prognostic_bc(grid::LatitudeLongitudeGrid{<:Any, <:Any, Flat}, ::Val{:south}, loc, default) = default
 default_auxiliary_bc(grid::LatitudeLongitudeGrid{<:Any, <:Any, Flat}, ::Val{:north}, loc) = nothing
 default_auxiliary_bc(grid::LatitudeLongitudeGrid{<:Any, <:Any, Flat}, ::Val{:south}, loc) = nothing
