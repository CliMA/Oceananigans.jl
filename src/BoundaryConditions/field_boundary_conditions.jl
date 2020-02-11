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

default_bc(::Periodic) = PeriodicBC()
default_bc(::Bounded)  = NoFluxBC()

valid_directions = (:west, :east, :south, :north, :bottom, :top)

function FieldBoundaryConditions(grid; kwargs...)
    kws = keys(kwargs)
    for kw in kws
        if kw ∉ valid_directions
            e = "$kw is not a valid keyword argument. Must be one of " *
                "the valid directions: $valid_directions"
            throw(ArgumentError(e))
        end
    end

    TX, TY, TZ = topology(grid)

    if (TX isa Periodic || TX isa Flat) && (:west in kws || :east in kws)
        e = "Cannot specify west or east boundary conditions with $TX topology in x-direction."
        throw(ArgumentError(e))
    else
        west_bc = :west in kws ? kwargs[:west] : default_bc(TX)
        east_bc = :east in kws ? kwargs[:east] : default_bc(TX)
    end

    if (TY isa Periodic || TY isa Flat) && (:south in kws || :north in kws)
        e = "Cannot specify south or north boundary conditions with $TY topology in y-direction."
        throw(ArgumentError(e))
    else
        south_bc = :south in kws ? kwargs[:south] : default_bc(TY)
        north_bc = :north in kws ? kwargs[:north] : default_bc(TY)
    end

    if (TZ isa Periodic || TZ isa Flat) && (:bottom in kws || :top in kws)
        e = "Cannot specify bottom or top boundary conditions with $TZ topology in z-direction."
        throw(ArgumentError(e))
    else
        bottom_bc = :bottom in kws ? kwargs[:bottom] : default_bc(TZ)
        top_bc    = :top    in kws ? kwargs[:top]    : default_bc(TZ)
    end

    x = CoordinateBoundaryConditions(west_bc, east_bc)
    y = CoordinateBoundaryConditions(south_bc, north_bc)
    z = CoordinateBoundaryConditions(bottom_bc, top_bc)

    return FieldBoundaryConditions(x, y, z)
end
