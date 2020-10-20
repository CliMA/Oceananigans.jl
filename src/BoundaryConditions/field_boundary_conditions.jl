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
    DefaultBoundaryCondition(::Type{Periodic}, loc)

Returns [`PeriodicBoundaryCondition`](@ref).
"""
DefaultBoundaryCondition(::Type{Grids.Periodic}, loc) = PeriodicBoundaryCondition()

"""
    DefaultBoundaryCondition(::Type{Flat}, loc)

Returns `nothing`.
"""
DefaultBoundaryCondition(::Type{Flat}, loc) = nothing

"""
    DefaultBoundaryCondition(::Type{Bounded}, ::Type{Cell})

Returns [`NoFluxBoundaryCondition`](@ref).
"""
DefaultBoundaryCondition(::Type{Bounded}, ::Type{Cell}) = NoFluxBoundaryCondition()

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

assign_location(bc, args...) = bc # fallback

"""
    FieldBoundaryConditions(grid, loc;   east = DefaultBoundaryCondition(topology(grid, 1), loc[1]),
                                         west = DefaultBoundaryCondition(topology(grid, 1], loc[1]),
                                        south = DefaultBoundaryCondition(topology(grid, 2), loc[2]),
                                        north = DefaultBoundaryCondition(topology(grid, 2), loc[2]),
                                       bottom = DefaultBoundaryCondition(topology(grid, 3), loc[3]),
                                          top = DefaultBoundaryCondition(topology(grid, 3), loc[3]))

Construct `FieldBoundaryConditions` for a field with location `loc` (a 3-tuple of `Face` or `Cell`)
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

    east   = assign_location(east,   Nothing, loc[2], loc[3])
    west   = assign_location(west,   Nothing, loc[2], loc[3])
    south  = assign_location(south,  loc[1], Nothing, loc[3])
    north  = assign_location(north,  loc[1], Nothing, loc[3])
    bottom = assign_location(bottom, loc[1], loc[2], Nothing)
    top    = assign_location(top,    loc[1], loc[2], Nothing)

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
                                       south = DefaultBoundaryCondition(topology(grid, 2), Cell),
                                       north = DefaultBoundaryCondition(topology(grid, 2), Cell),
                                      bottom = DefaultBoundaryCondition(topology(grid, 3), Cell),
                                         top = DefaultBoundaryCondition(topology(grid, 3), Cell))

Construct `FieldBoundaryConditions` for the `u`-velocity field on `grid`.
Boundary conditions on `x`-, `y`-, and `z`-boundaries are specified via
keyword arguments:

    * `west` and `east` for the `-x` and `+x` boundary;
    * `south` and `north` for the `-y` and `+y` boundary;
    * `bottom` and `top` for the `-z` and `+z` boundary.

Default boundary conditions depend on `topology(grid)`. See `DefaultBoundaryCondition`.
"""
UVelocityBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Face, Cell, Cell); user_defined_bcs...)

"""
    VVelocityBoundaryConditions(grid;   east = DefaultBoundaryCondition(topology(grid, 1), Cell),
                                        west = DefaultBoundaryCondition(topology(grid, 1), Cell),
                                       south = DefaultBoundaryCondition(topology(grid, 2), Face),
                                       north = DefaultBoundaryCondition(topology(grid, 2), Face),
                                      bottom = DefaultBoundaryCondition(topology(grid, 3), Cell),
                                         top = DefaultBoundaryCondition(topology(grid, 3), Cell))

Construct `FieldBoundaryConditions` for the `v`-velocity field on `grid`.
Boundary conditions on `x`-, `y`-, and `z`-boundaries are specified via
keyword arguments:

    * `west` and `east` for the `-x` and `+x` boundary;
    * `south` and `north` for the `-y` and `+y` boundary;
    * `bottom` and `top` for the `-z` and `+z` boundary.

Default boundary conditions depend on `topology(grid)`. See `DefaultBoundaryCondition`.
"""
VVelocityBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Cell, Face, Cell); user_defined_bcs...)

"""
    WVelocityBoundaryConditions(grid;   east = DefaultBoundaryCondition(topology(grid, 1), Cell),
                                        west = DefaultBoundaryCondition(topology(grid, 1), Cell),
                                       south = DefaultBoundaryCondition(topology(grid, 2), Cell),
                                       north = DefaultBoundaryCondition(topology(grid, 2), Cell),
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
WVelocityBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Cell, Cell, Face); user_defined_bcs...)

"""
    TracerBoundaryConditions(grid;   east = DefaultBoundaryCondition(topology(grid, 1), Cell),
                                     west = DefaultBoundaryCondition(topology(grid, 1), Cell),
                                    south = DefaultBoundaryCondition(topology(grid, 2), Cell),
                                    north = DefaultBoundaryCondition(topology(grid, 2), Cell),
                                   bottom = DefaultBoundaryCondition(topology(grid, 3), Cell),
                                      top = DefaultBoundaryCondition(topology(grid, 3), Cell))

Construct `FieldBoundaryConditions` for a tracer field on `grid`.
Boundary conditions on `x`-, `y`-, and `z`-boundaries are specified via
keyword arguments:

    * `west` and `east` for the `-x` and `+x` boundary;
    * `south` and `north` for the `-y` and `+y` boundary;
    * `bottom` and `top` for the `-z` and `+z` boundary.

Default boundary conditions depend on `topology(grid)`. See `DefaultBoundaryCondition`.
"""
TracerBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Cell, Cell, Cell); user_defined_bcs...)

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

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions, grid, field_name, tracer_names)
    model_field_names = tuple(:u, :v, :w, tracer_names...)

    east   = regularize_boundary_condition(bcs.east,   grid.Nx, field_name, model_field_names)
    west   = regularize_boundary_condition(bcs.west,   1,       field_name, model_field_names)
    north  = regularize_boundary_condition(bcs.north,  grid.Ny, field_name, model_field_names)
    south  = regularize_boundary_condition(bcs.south,  1,       field_name, model_field_names)
    top    = regularize_boundary_condition(bcs.top,    grid.Nz, field_name, model_field_names)
    bottom = regularize_boundary_condition(bcs.bottom, 1,       field_name, model_field_names)

    x = CoordinateBoundaryConditions(west, east)
    y = CoordinateBoundaryConditions(south, north)
    z = CoordinateBoundaryConditions(bottom, top)

    return FieldBoundaryConditions(x, y, z)
end
