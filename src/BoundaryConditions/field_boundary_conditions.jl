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

DefaultBoundaryCondition(::Union{Grids.Periodic, Flat}, loc) = PeriodicBoundaryCondition()

DefaultBoundaryCondition(::Bounded, ::Cell) = NoFluxBoundaryCondition()
DefaultBoundaryCondition(::Bounded, ::Face) = NoPenetrationBoundaryCondition()

function validate_bcs(topology, left_bc, right_bc, default_bc, left_name, right_name, dir)
    if topology isa Periodic && (left_bc != default_bc || right_bc != default_bc)
        e = "Cannot specify $left_name or $right_name boundary conditions with " *
            "$topology topology in $dir-direction."
        throw(ArgumentError(e))
    end
    return true
end

"""
    FieldBoundaryConditions(grid, loc;
          east = DefaultBoundaryCondition(topology(grid)[1], loc[1]),
          west = DefaultBoundaryCondition(topology(grid)[1], loc[1]),
         south = DefaultBoundaryCondition(topology(grid)[2], loc[2]),
         north = DefaultBoundaryCondition(topology(grid)[2], loc[2]),
        bottom = DefaultBoundaryCondition(topology(grid)[3], loc[3]),
           top = DefaultBoundaryCondition(topology(grid)[3], loc[3]))

Construct `FieldBoundaryConditions` for a field with location `loc` (a 3-tuple of `Face` or `Cell`)
defined on `grid` (the grid's topology is what defined the default boundary conditions that are
imposed).

Specific boundary conditions can be applied along the x dimension with the `west` and `east` kwargs,
along the y-dimension with the `south` and `north` kwargs, and along the z-dimension with the `bottom`
and `top` kwargs.
"""
function FieldBoundaryConditions(grid, loc;
      east = DefaultBoundaryCondition(topology(grid)[1], loc[1]),
      west = DefaultBoundaryCondition(topology(grid)[1], loc[1]),
     south = DefaultBoundaryCondition(topology(grid)[2], loc[2]),
     north = DefaultBoundaryCondition(topology(grid)[2], loc[2]),
    bottom = DefaultBoundaryCondition(topology(grid)[3], loc[3]),
       top = DefaultBoundaryCondition(topology(grid)[3], loc[3]))

    TX, TY, TZ = topology(grid)
    x_default_bc = DefaultBoundaryCondition(topology(grid)[1], loc[1])
    y_default_bc = DefaultBoundaryCondition(topology(grid)[2], loc[2])
    z_default_bc = DefaultBoundaryCondition(topology(grid)[3], loc[3])

    validate_bcs(TX, west,   east, x_default_bc, :west,   :east, :x)
    validate_bcs(TY, south, north, y_default_bc, :south, :north, :y)
    validate_bcs(TZ, bottom,  top, z_default_bc, :bottom,  :top, :z)

    x = CoordinateBoundaryConditions(west, east)
    y = CoordinateBoundaryConditions(south, north)
    z = CoordinateBoundaryConditions(bottom, top)

    return FieldBoundaryConditions(x, y, z)
end

  UVelocityBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Face(), Cell(), Cell()); user_defined_bcs...)
  VVelocityBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Cell(), Face(), Cell()); user_defined_bcs...)
  WVelocityBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Cell(), Cell(), Face()); user_defined_bcs...)
     TracerBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Cell(), Cell(), Cell()); user_defined_bcs...)
   PressureBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Cell(), Cell(), Cell()); user_defined_bcs...)
DiffusivityBoundaryConditions(grid; user_defined_bcs...) = FieldBoundaryConditions(grid, (Cell(), Cell(), Cell()); user_defined_bcs...)
