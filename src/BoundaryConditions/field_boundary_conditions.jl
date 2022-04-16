using Oceananigans.Operators: assumed_field_location

#####
##### Default boundary conditions
#####

struct DefaultPrognosticFieldBoundaryCondition end

default_prognostic_bc(::Grids.Periodic, loc)       = PeriodicBoundaryCondition()
default_prognostic_bc(::Bounded,        ::Center)  = NoFluxBoundaryCondition()
default_prognostic_bc(::Bounded,        ::Face)    = ImpenetrableBoundaryCondition()

default_prognostic_bc(::Bounded,        ::Nothing) = nothing
default_prognostic_bc(::Flat,           ::Nothing) = nothing
default_prognostic_bc(::Grids.Periodic, ::Nothing) = nothing
default_prognostic_bc(::Flat, loc) = nothing

default_auxiliary_bc(topo, loc) = default_prognostic_bc(topo, loc)
default_auxiliary_bc(::Bounded, ::Face) = nothing

#####
##### Field boundary conditions
#####

mutable struct FieldBoundaryConditions{W, E, S, N, B, T, I}
        west :: W
        east :: E
       south :: S
       north :: N
      bottom :: B
         top :: T
    immersed :: I
end

function FieldBoundaryConditions(indices::Tuple, west, east, south, north, bottom, top, immersed)
    # Turn bcs in windowed dimensions into nothing
    west, east   = window_boundary_conditions(indices[1], west, east)
    south, north = window_boundary_conditions(indices[2], south, north)
    bottom, top  = window_boundary_conditions(indices[3], bottom, top)
    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

FieldBoundaryConditions(indices::Tuple, bcs::FieldBoundaryConditions) =
    FieldBoundaryConditions(indices, (getproperty(bcs, side) for side in fieldnames(FieldBoundaryConditions))...)

window_boundary_conditions(::Colon, left, right) = left, right
window_boundary_conditions(::UnitRange, left, right) = nothing, nothing

"""
    FieldBoundaryConditions(; kwargs...)

Return a template for boundary conditions on prognostic fields.

Keyword arguments
=================

Keyword arguments specify boundary conditions on the 7 possible boundaries:

  - `west`, left end point in the `x`-direction where `i=1`
  - `east`, right end point in the `x`-direction where `i=grid.Nx`
  - `south`, left end point in the `y`-direction where `j=1`
  - `north`, right end point in the `y`-direction where `j=grid.Ny`
  - `bottom`, right end point in the `z`-direction where `k=1`
  - `top`, right end point in the `z`-direction where `k=grid.Nz`
  - `immersed`, boundary between solid and fluid for immersed boundaries (experimental support only)

If a boundary condition is unspecified, the default for prognostic fields
and the topology in the boundary-normal direction is used:

  - `PeriodicBoundaryCondition` for `Periodic` directions
  - `NoFluxBoundaryCondition` for `Bounded` directions and `Centered`-located fields
  - `ImpenetrableBoundaryCondition` for `Bounded` directions and `Face`-located fields
  - `nothing` for `Flat` directions and/or `Nothing`-located fields
"""
function FieldBoundaryConditions(;     west = DefaultPrognosticFieldBoundaryCondition(),
                                       east = DefaultPrognosticFieldBoundaryCondition(),
                                      south = DefaultPrognosticFieldBoundaryCondition(),
                                      north = DefaultPrognosticFieldBoundaryCondition(),
                                     bottom = DefaultPrognosticFieldBoundaryCondition(),
                                        top = DefaultPrognosticFieldBoundaryCondition(),
                                   immersed = NoFluxBoundaryCondition())

   return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

"""
    FieldBoundaryConditions(grid, location; kwargs...)

Return boundary conditions for auxiliary fields (fields whose values are
derived from a model's prognostic fields) on `grid` and at `location`.

Keyword arguments
=================

Keyword arguments specify boundary conditions on the 6 possible boundaries:

  - `west`, left end point in the `x`-direction where `i=1`
  - `east`, right end point in the `x`-direction where `i=grid.Nx`
  - `south`, left end point in the `y`-direction where `j=1`
  - `north`, right end point in the `y`-direction where `j=grid.Ny`
  - `bottom`, right end point in the `z`-direction where `k=1`
  - `top`, right end point in the `z`-direction where `k=grid.Nz`

If a boundary condition is unspecified, the default for auxiliary fields
and the topology in the boundary-normal direction is used:

  - `PeriodicBoundaryCondition` for `Periodic` directions
  - `GradientBoundaryCondition(0)` for `Bounded` directions and `Centered`-located fields
  - `nothing` for `Bounded` directions and `Face`-located fields
  - `nothing` for `Flat` directions and/or `Nothing`-located fields)
"""
function FieldBoundaryConditions(grid, loc, indices=(:, :, :);
                                 west     = default_auxiliary_bc(topology(grid, 1)(), loc[1]()),
                                 east     = default_auxiliary_bc(topology(grid, 1)(), loc[1]()),
                                 south    = default_auxiliary_bc(topology(grid, 2)(), loc[2]()),
                                 north    = default_auxiliary_bc(topology(grid, 2)(), loc[2]()),
                                 bottom   = default_auxiliary_bc(topology(grid, 3)(), loc[3]()),
                                 top      = default_auxiliary_bc(topology(grid, 3)(), loc[3]()),
                                 immersed = NoFluxBoundaryCondition())
    
    return FieldBoundaryConditions(indices, west, east, south, north, bottom, top, immersed)
end

#####
##### Boundary condition "regularization"
#####

regularize_immersed_boundary_condition(ibc::ZFBC, args...) = ibc # default

# Friendly warning?
function regularize_immersed_boundary_condition(ibc, grid, loc, field_name, args...)
    msg = """
          $field_name was assigned an immersed $ibc, but this is not supported on
          $(summary(grid))
          The immersed boundary condition on $field_name will have no effect.
          """

    @warn msg

    return nothing
end

regularize_boundary_condition(::DefaultPrognosticFieldBoundaryCondition, topo, loc, dim, args...) =
    default_prognostic_bc(topo[dim](), loc[dim]())

regularize_boundary_condition(bc, args...) = bc # fallback

""" 
Compute default boundary conditions and attach field locations to ContinuousBoundaryFunction
boundary conditions for prognostic model field boundary conditions.

!!! warn "No support for `ContinuousBoundaryFunction` for immersed boundary conditions"
    Do not regularize immersed boundary conditions.

    Currently, there is no support `ContinuousBoundaryFunction` for immersed boundary
    conditions.
"""
function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                              grid::AbstractGrid,
                                              field_name::Symbol,
                                              prognostic_names=nothing)
    topo = topology(grid)
    loc = assumed_field_location(field_name)
    
    west     = regularize_boundary_condition(bcs.west,   topo, loc, 1, LeftBoundary,  prognostic_names)
    east     = regularize_boundary_condition(bcs.east,   topo, loc, 1, RightBoundary, prognostic_names)
    south    = regularize_boundary_condition(bcs.south,  topo, loc, 2, LeftBoundary,  prognostic_names)
    north    = regularize_boundary_condition(bcs.north,  topo, loc, 2, RightBoundary, prognostic_names)
    bottom   = regularize_boundary_condition(bcs.bottom, topo, loc, 3, LeftBoundary,  prognostic_names)
    top      = regularize_boundary_condition(bcs.top,    topo, loc, 3, RightBoundary, prognostic_names)

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

