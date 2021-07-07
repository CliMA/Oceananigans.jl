using Oceananigans.Operators: assumed_field_location

struct DefaultBoundaryCondition end

"""
    FieldBoundaryConditions

An alias for `NamedTuple{(:x, :y, :z)}` that represents a set of three `CoordinateBoundaryCondition`s
applied to a field along x, y, and z.
"""
struct FieldBoundaryConditions{W, E, S, N, B, T, I}
    west :: W
    east :: E
    south :: S
    north :: N
    bottom :: B
    top :: T
    immersed :: I
end

function FieldBoundaryConditions(; east = DefaultBoundaryCondition(),
                                   west = DefaultBoundaryCondition(),
                                   south = DefaultBoundaryCondition(),
                                   north = DefaultBoundaryCondition(),
                                   bottom = DefaultBoundaryCondition(),
                                   top = DefaultBoundaryCondition(),
                                   immersed = DefaultBoundaryCondition())

   return FieldBoundaryConditions(east, west, south, north, bottom, top, immersed)
end

"""
    regularize_boundary_condition(topo, ::Type{Nothing})

Returns nothing.
"""
regularize_boundary_condition(::DefaultBoundaryCondition, LX, LY, LZ, topo, ::Type{Nothing}) = nothing
regularize_boundary_condition(::Type{Grids.Periodic}, ::Type{Nothing}) = nothing

"""
    regularize_boundary_condition(::Type{Periodic}, loc)

Returns [`PeriodicBoundaryCondition`](@ref).
"""
regularize_boundary_condition(::Type{Grids.Periodic}, loc) = PeriodicBoundaryCondition()

"""
    regularize_boundary_condition(::Type{Flat}, loc)

Returns `nothing`.
"""
regularize_boundary_condition(::Type{Flat}, loc) = nothing
regularize_boundary_condition(::Type{Flat}, ::Type{Nothing}) = nothing

"""
    regularize_boundary_condition(::Type{Bounded}, ::Type{Center})

Returns [`NoFluxBoundaryCondition`](@ref).
"""
regularize_boundary_condition(::Type{Bounded}, ::Type{Center}) = NoFluxBoundaryCondition()

"""
    regularize_boundary_condition(::Type{Bounded}, ::Type{Face})

Returns [`ImpenetrableBoundaryCondition`](@ref).
"""
regularize_boundary_condition(::Type{Bounded}, ::Type{Face}) = ImpenetrableBoundaryCondition()

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions, grid, model_field_names, field_name)

    X, Y, Z = assumed_field_location(field_name)

    east   = regularize_boundary_condition(bcs.east,   Nothing, Y, Z, grid.Nx, model_field_names)
    west   = regularize_boundary_condition(bcs.west,   Nothing, Y, Z, 1,       model_field_names)
    north  = regularize_boundary_condition(bcs.north,  X, Nothing, Z, grid.Ny, model_field_names)
    south  = regularize_boundary_condition(bcs.south,  X, Nothing, Z, 1,       model_field_names)
    top    = regularize_boundary_condition(bcs.top,    X, Y, Nothing, grid.Nz, model_field_names)
    bottom = regularize_boundary_condition(bcs.bottom, X, Y, Nothing, 1,       model_field_names)
    immersed = regularize_boundary_condition(bcs.bottom, X, Y, Nothing, 1,       model_field_names)

    return FieldBoundaryConditions(east, west, north, south, top, bottom, immersed)
end

function regularize_field_boundary_conditions(boundary_conditions::NamedTuple, grid, model_field_names, field_name=nothing)
    boundary_conditions_names = propertynames(boundary_conditions)
    boundary_conditions_tuple = Tuple(regularize_field_boundary_conditions(bcs, grid, model_field_names, name)
                                      for (name, bcs) in zip(boundary_conditions_names, boundary_conditions))
    boundary_conditions = NamedTuple{boundary_conditions_names}(boundary_conditions_tuple)
    return boundary_conditions
end

regularize_field_boundary_conditions(::Missing, grid, model_field_names, field_name=nothing) = missing
