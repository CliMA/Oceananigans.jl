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

default_bc(::Grids.Periodic) = PeriodicBC()  # To avoid conflict with BoundaryConditions.Periodic
default_bc(::Bounded)  = NoFluxBC()
default_bc(::Flat)     = PeriodicBC()

default_x_bc(grid) = default_bc(topology(grid)[1])
default_y_bc(grid) = default_bc(topology(grid)[2])
default_z_bc(grid) = default_bc(topology(grid)[3])

function validate_bcs(topology, left_bc, right_bc, default_bc, left_name, right_name, dir)
    if topology isa Periodic && (left_bc != default_bc || right_bc != default_bc)
        e = "Cannot specify $left_name or $right_name boundary conditions with $topology topology in $dir-direction."
        throw(ArgumentError(e))
    end
    return true
end

function FieldBoundaryConditions(grid::AbstractGrid; west=default_x_bc(grid), east=default_x_bc(grid),
                                 south=default_y_bc(grid), north=default_y_bc(grid),
                                 bottom=default_z_bc(grid), top=default_z_bc(grid))
    TX, TY, TZ = topology(grid)
    validate_bcs(TX, west,   east, default_x_bc(grid), :west,   :east, :x)
    validate_bcs(TY, south, north, default_y_bc(grid), :south, :north, :y)
    validate_bcs(TZ, bottom,  top, default_z_bc(grid), :bottom,  :top, :z)

    x = CoordinateBoundaryConditions(west, east)
    y = CoordinateBoundaryConditions(south, north)
    z = CoordinateBoundaryConditions(bottom, top)

    return FieldBoundaryConditions(x, y, z)
end
