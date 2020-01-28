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
    HorizontallyPeriodicBCs(;   top = BoundaryCondition(Flux, nothing),
                             bottom = BoundaryCondition(Flux, nothing))

Construct `FieldBoundaryConditions` with `Periodic` boundary conditions in the x and y
directions and specified `top` (+z) and `bottom` (-z) boundary conditions for u, v,
and tracer fields.

`HorizontallyPeriodicBCs` cannot be applied to the the vertical velocity w.
"""
function HorizontallyPeriodicBCs(;    top = BoundaryCondition(Flux, nothing),
                                   bottom = BoundaryCondition(Flux, nothing))

    x = PeriodicBCs()
    y = PeriodicBCs()
    z = CoordinateBoundaryConditions(bottom, top)

    return FieldBoundaryConditions(x, y, z)
end

"""
    ChannelBCs(; north = BoundaryCondition(Flux, nothing),
                 south = BoundaryCondition(Flux, nothing),
                   top = BoundaryCondition(Flux, nothing),
                bottom = BoundaryCondition(Flux, nothing))

Construct `FieldBoundaryConditions` with `Periodic` boundary conditions in the x
direction and specified `north` (+y), `south` (-y), `top` (+z) and `bottom` (-z)
boundary conditions for u, v, and tracer fields.

`ChannelBCs` cannot be applied to the the vertical velocity w.
"""
function ChannelBCs(;  north = BoundaryCondition(Flux, nothing),
                       south = BoundaryCondition(Flux, nothing),
                         top = BoundaryCondition(Flux, nothing),
                      bottom = BoundaryCondition(Flux, nothing)
                    )

    x = PeriodicBCs()
    y = CoordinateBoundaryConditions(south, north)
    z = CoordinateBoundaryConditions(bottom, top)

    return FieldBoundaryConditions(x, y, z)
end
