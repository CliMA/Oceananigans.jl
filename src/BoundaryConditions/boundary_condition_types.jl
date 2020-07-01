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
    NormalFlow

A type specifying the component of a velocity field normal to a boundary.

Thus `NormalFlow` can only be applied to `u` along x, `v` along y, or `w` along z.
For all other cases --- fields located at (Cell, Cell, Cell), or `u`, `v`,
and `w` in (y, z), (x, z), and (x, y), respectively, either `Value`,
`Gradient`, or `Flux` conditions must be used.

Note that `NormalFlow` differs from a zero `Value` boundary condition: 
`Value` imposes values at cell centers, while `NormalFlow` imposes values
_on_ a boundary, at cell faces. Only wall-normal components of the velocity field are defined
on cell faces with respect to the wall-normal direction, and therefore only wall-normal
components of the velocity field are defined on boundaries. 
Both tracers and wall-tangential components of velocity fields
are defined at cell centers with respect to the wall-normal direction.
"""
struct NormalFlow <: BCType end
