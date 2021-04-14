using Oceananigans.Operators: assumed_field_location

"""
    FieldBoundaryConditions

An alias for `NamedTuple{(:x, :y, :z)}` that represents a set of three `CoordinateBoundaryCondition`s
applied to a field along x, y, and z.
"""
const FieldBoundaryConditions = NamedTuple{(:x, :y, :z)}

"""
    FieldBoundaryConditions(x, y, z)

Construct a `FieldBoundaryConditions` using a `CoordinateBoundaryCondition` for each of the
`x`, `y`, and `z` coordinates.
"""
FieldBoundaryConditions(x, y, z) = FieldBoundaryConditions((x, y, z))

"""
    DefaultBoundaryCondition(topo, ::Type{Nothing})

Returns nothing.
"""
DefaultBoundaryCondition(topo, ::Type{Nothing}) = nothing
DefaultBoundaryCondition(::Type{Grids.Periodic}, ::Type{Nothing}) = nothing

"""
    DefaultBoundaryCondition(::Type{Periodic}, loc)

Returns [`PeriodicBoundaryCondition`](@ref).
"""
DefaultBoundaryCondition(::Type{Grids.Periodic}, loc) = PeriodicBoundaryCondition()

"""
    DefaultBoundaryCondition(::Type{Flat}, loc)

Returns `nothing`.
"""
DefaultBoundaryCondition(::Type{Flat}, loc) = nothing
DefaultBoundaryCondition(::Type{Flat}, ::Type{Nothing}) = nothing

"""
    DefaultBoundaryCondition(::Type{Bounded}, ::Type{Center})

Returns [`NoFluxBoundaryCondition`](@ref).
"""
DefaultBoundaryCondition(::Type{Bounded}, ::Type{Center}) = NoFluxBoundaryCondition()

"""
    DefaultBoundaryCondition(::Type{Bounded}, ::Type{Face})

Returns [`ImpenetrableBoundaryCondition`](@ref).
"""
DefaultBoundaryCondition(::Type{Bounded}, ::Type{Face}) = ImpenetrableBoundaryCondition()

function validate_bcs(topology, left_bc, right_bc, default_bc, left_name, right_name, dir)

    if topology isa Periodic && (left_bc != default_bc || right_bc != default_bc)
        e = "Cannot specify $left_name or $right_name boundary conditions with " *
            "$topology topology in $dir-direction."
        throw(ArgumentError(e))
    end

    return true
end

"""
    FieldBoundaryConditions(grid, loc;   east = DefaultBoundaryCondition(topology(grid, 1), loc[1]),
                                         west = DefaultBoundaryCondition(topology(grid, 1], loc[1]),
                                        south = DefaultBoundaryCondition(topology(grid, 2), loc[2]),
                                        north = DefaultBoundaryCondition(topology(grid, 2), loc[2]),
                                       bottom = DefaultBoundaryCondition(topology(grid, 3), loc[3]),
                                          top = DefaultBoundaryCondition(topology(grid, 3), loc[3]))

Construct `FieldBoundaryConditions` for a field with location `loc` (a 3-tuple of `Face` or `Center`)
defined on `grid`.

Boundary conditions on `x`-, `y`-, and `z`-boundaries are specified via
keyword arguments:

    * `west` and `east` for the `-x` and `+x` boundary;
    * `south` and `north` for the `-y` and `+y` boundary;
    * `bottom` and `top` for the `-z` and `+z` boundary.

Default boundary conditions depend on `topology(grid)` and `loc`.
"""
function FieldBoundaryConditions(grid, loc;   east = DefaultBoundaryCondition(topology(grid, 1), loc[1]),
                                              west = DefaultBoundaryCondition(topology(grid, 1), loc[1]),
                                             south = DefaultBoundaryCondition(topology(grid, 2), loc[2]),
                                             north = DefaultBoundaryCondition(topology(grid, 2), loc[2]),
                                            bottom = DefaultBoundaryCondition(topology(grid, 3), loc[3]),
                                               top = DefaultBoundaryCondition(topology(grid, 3), loc[3]))

    TX, TY, TZ = topology(grid)
    x_default_bc = DefaultBoundaryCondition(topology(grid, 1), loc[1])
    y_default_bc = DefaultBoundaryCondition(topology(grid, 2), loc[2])
    z_default_bc = DefaultBoundaryCondition(topology(grid, 3), loc[3])

    validate_bcs(TX, west,   east, x_default_bc, :west,   :east, :x)
    validate_bcs(TY, south, north, y_default_bc, :south, :north, :y)
    validate_bcs(TZ, bottom,  top, z_default_bc, :bottom,  :top, :z)

    x = CoordinateBoundaryConditions(west, east)
    y = CoordinateBoundaryConditions(south, north)
    z = CoordinateBoundaryConditions(bottom, top)

    return FieldBoundaryConditions(x, y, z)
end

"""
    UVelocityBoundaryConditions(grid;   east = DefaultBoundaryCondition(topology(grid, 1), Face),
                                        west = DefaultBoundaryCondition(topology(grid, 1), Face),
                                       south = DefaultBoundaryCondition(topology(grid, 2), Center),
                                       north = DefaultBoundaryCondition(topology(grid, 2), Center),
                                      bottom = DefaultBoundaryCondition(topology(grid, 3), Center),
                                         top = DefaultBoundaryCondition(topology(grid, 3), Center))

Construct `FieldBoundaryConditions` for the `u`-velocity field on `grid`.
Boundary conditions on `x`-, `y`-, and `z`-boundaries are specified via
keyword arguments:

    * `west` and `east` for the `-x` and `+x` boundary;
    * `south` and `north` for the `-y` and `+y` boundary;
    * `bottom` and `top` for the `-z` and `+z` boundary.

Default boundary conditions depend on `topology(grid)`. See `DefaultBoundaryCondition`.
"""
UVelocityBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Face, Center, Center); user_defined_bcs...)

"""
    VVelocityBoundaryConditions(grid;   east = DefaultBoundaryCondition(topology(grid, 1), Center),
                                        west = DefaultBoundaryCondition(topology(grid, 1), Center),
                                       south = DefaultBoundaryCondition(topology(grid, 2), Face),
                                       north = DefaultBoundaryCondition(topology(grid, 2), Face),
                                      bottom = DefaultBoundaryCondition(topology(grid, 3), Center),
                                         top = DefaultBoundaryCondition(topology(grid, 3), Center))

Construct `FieldBoundaryConditions` for the `v`-velocity field on `grid`.
Boundary conditions on `x`-, `y`-, and `z`-boundaries are specified via
keyword arguments:

    * `west` and `east` for the `-x` and `+x` boundary;
    * `south` and `north` for the `-y` and `+y` boundary;
    * `bottom` and `top` for the `-z` and `+z` boundary.

Default boundary conditions depend on `topology(grid)`. See `DefaultBoundaryCondition`.
"""
VVelocityBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Center, Face, Center); user_defined_bcs...)

"""
    WVelocityBoundaryConditions(grid;   east = DefaultBoundaryCondition(topology(grid, 1), Center),
                                        west = DefaultBoundaryCondition(topology(grid, 1), Center),
                                       south = DefaultBoundaryCondition(topology(grid, 2), Center),
                                       north = DefaultBoundaryCondition(topology(grid, 2), Center),
                                      bottom = DefaultBoundaryCondition(topology(grid, 3), Face),
                                         top = DefaultBoundaryCondition(topology(grid, 3), Face))

Construct `FieldBoundaryConditions` for the `w`-velocity field on `grid`.
Boundary conditions on `x`-, `y`-, and `z`-boundaries are specified via
keyword arguments:

    * `west` and `east` for the `-x` and `+x` boundary;
    * `south` and `north` for the `-y` and `+y` boundary;
    * `bottom` and `top` for the `-z` and `+z` boundary.

Default boundary conditions depend on `topology(grid)`. See `DefaultBoundaryCondition`.
"""
WVelocityBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Center, Center, Face); user_defined_bcs...)

"""
    TracerBoundaryConditions(grid;   east = DefaultBoundaryCondition(topology(grid, 1), Center),
                                     west = DefaultBoundaryCondition(topology(grid, 1), Center),
                                    south = DefaultBoundaryCondition(topology(grid, 2), Center),
                                    north = DefaultBoundaryCondition(topology(grid, 2), Center),
                                   bottom = DefaultBoundaryCondition(topology(grid, 3), Center),
                                      top = DefaultBoundaryCondition(topology(grid, 3), Center))

Construct `FieldBoundaryConditions` for a tracer field on `grid`.
Boundary conditions on `x`-, `y`-, and `z`-boundaries are specified via
keyword arguments:

    * `west` and `east` for the `-x` and `+x` boundary;
    * `south` and `north` for the `-y` and `+y` boundary;
    * `bottom` and `top` for the `-z` and `+z` boundary.

Default boundary conditions depend on `topology(grid)`. See `DefaultBoundaryCondition`.
"""
TracerBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Center, Center, Center); user_defined_bcs...)

const PressureBoundaryConditions = TracerBoundaryConditions
const DiffusivityBoundaryConditions = TracerBoundaryConditions

# Here we overload setproperty! and getproperty to permit users to call
# the 'left' and 'right' bcs in the z-direction 'bottom' and 'top'
# and the 'left' and 'right' bcs in the y-direction 'south' and 'north'.
Base.setproperty!(bcs::FieldBoundaryConditions, side::Symbol, bc) = setbc!(bcs, Val(side), bc)

setbc!(bcs::FieldBoundaryConditions, ::Val{S}, bc) where S = setfield!(bcs, S, bc)
setbc!(bcs::FieldBoundaryConditions, ::Val{:west},   bc) = setfield!(bcs.x, :left,  bc)
setbc!(bcs::FieldBoundaryConditions, ::Val{:east},   bc) = setfield!(bcs.x, :right, bc)
setbc!(bcs::FieldBoundaryConditions, ::Val{:south},  bc) = setfield!(bcs.y, :left,  bc)
setbc!(bcs::FieldBoundaryConditions, ::Val{:north},  bc) = setfield!(bcs.y, :right, bc)
setbc!(bcs::FieldBoundaryConditions, ::Val{:bottom}, bc) = setfield!(bcs.z, :left,  bc)
setbc!(bcs::FieldBoundaryConditions, ::Val{:top},    bc) = setfield!(bcs.z, :right, bc)

@inline Base.getproperty(bcs::FieldBoundaryConditions, side::Symbol) = getbc(bcs, Val(side))

@inline getbc(bcs::FieldBoundaryConditions, ::Val{S}) where S = getfield(bcs, S)
@inline getbc(bcs::FieldBoundaryConditions, ::Val{:west})   = getfield(bcs.x, :left)
@inline getbc(bcs::FieldBoundaryConditions, ::Val{:east})   = getfield(bcs.x, :right)
@inline getbc(bcs::FieldBoundaryConditions, ::Val{:south})  = getfield(bcs.y, :left)
@inline getbc(bcs::FieldBoundaryConditions, ::Val{:north})  = getfield(bcs.y, :right)
@inline getbc(bcs::FieldBoundaryConditions, ::Val{:bottom}) = getfield(bcs.z, :left)
@inline getbc(bcs::FieldBoundaryConditions, ::Val{:top})    = getfield(bcs.z, :right)

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions, grid, model_field_names, field_name)

    X, Y, Z = assumed_field_location(field_name)

    east   = regularize_boundary_condition(bcs.east,   Nothing, Y, Z, grid.Nx, model_field_names)
    west   = regularize_boundary_condition(bcs.west,   Nothing, Y, Z, 1,       model_field_names)
    north  = regularize_boundary_condition(bcs.north,  X, Nothing, Z, grid.Ny, model_field_names)
    south  = regularize_boundary_condition(bcs.south,  X, Nothing, Z, 1,       model_field_names)
    top    = regularize_boundary_condition(bcs.top,    X, Y, Nothing, grid.Nz, model_field_names)
    bottom = regularize_boundary_condition(bcs.bottom, X, Y, Nothing, 1,       model_field_names)

    x = CoordinateBoundaryConditions(west, east)
    y = CoordinateBoundaryConditions(south, north)
    z = CoordinateBoundaryConditions(bottom, top)

    return FieldBoundaryConditions(x, y, z)
end

function regularize_field_boundary_conditions(boundary_conditions::NamedTuple, grid, model_field_names, field_name=nothing)
    boundary_conditions_names = propertynames(boundary_conditions)
    boundary_conditions_tuple = Tuple(regularize_field_boundary_conditions(bcs, grid, model_field_names, name)
                                      for (name, bcs) in zip(boundary_conditions_names, boundary_conditions))
    boundary_conditions = NamedTuple{boundary_conditions_names}(boundary_conditions_tuple)
    return boundary_conditions
end

regularize_field_boundary_conditions(::Missing, grid, model_field_names, field_name=nothing) = missing
