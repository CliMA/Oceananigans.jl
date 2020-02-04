"""
    BCType

Abstract supertype for boundary condition types.
"""
abstract type BCType end

"""
    Periodic

A type specifying a periodic boundary condition.

A condition may not be specified with a `Periodic` boundary condition.
"""
struct Periodic <: BCType end

"""
    Flux

A type specifying a boundary condition on the flux of a field.
"""
struct Flux <: BCType end

"""
    Gradient

A type specifying a boundary condition on the derivative or gradient of a field. Also
called a Neumann boundary condition.
"""
struct Gradient <: BCType end

"""
    Value

A type specifying a boundary condition on the value of a field. Also called a Dirchlet
boundary condition.
"""
struct Value <: BCType end

"""
    NoPenetration

A type specifying a no-penetration boundary condition for a velocity component that is normal to a wall.

Thus `NoPenetration` can only be applied to `u` along x, `v` along y, or `w` along z.
For all other cases --- fields located at (Cell, Cell, Cell), or `u`, `v`,
and `w` in (y, z), (x, z), and (x, y), respectively, either `Value`,
`Gradient`, or `Flux` conditions must be used.

A condition may not be specified with a `NoPenetration` boundary condition.

Note that this differs from a zero `Value` boundary condition as `Value` imposes values at the cell centers
(and could apply to tracers) while a no-penetration boundary condition only applies
to normal velocity components at a wall, where the velocity at the cell face collocated
at the wall is known and set to zero.
"""
struct NoPenetration <: BCType end
