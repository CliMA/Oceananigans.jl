"""
    AbstractBoundaryConditionClassification

Abstract supertype for boundary condition types.
"""
abstract type AbstractBoundaryConditionClassification end

"""
    Periodic

A type specifying a periodic boundary condition.

A condition may not be specified with a `Periodic` boundary condition.
"""
struct Periodic <: AbstractBoundaryConditionClassification end

"""
    Flux

A type specifying a boundary condition on the flux of a field.

The sign convention is such that a positive flux represents the flux of a quantity in the
positive direction. For example, a positive vertical flux implies a quantity is fluxed
upwards, in the +z direction.

Due to this convention, a positive flux applied to the top boundary specifies that a quantity
is fluxed upwards across the top boundary and thus out of the domain. As a result, a positive
flux applied to a top boundary leads to a reduction of that quantity in the interior of the
domain; for example, a positive, upwards flux of heat at the top of the domain acts to cool
the interior of the domain. Conversely, a positive flux applied to the bottom boundary leads
to an increase of the quantity in the interior of the domain. The same logic holds for east,
west, north, and south boundaries.
"""
struct Flux <: AbstractBoundaryConditionClassification end

"""
    Gradient

A type specifying a boundary condition on the derivative or gradient of a field. Also
called a Neumann boundary condition.
"""
struct Gradient <: AbstractBoundaryConditionClassification end

"""
    Value

A type specifying a boundary condition on the value of a field. Also called a Dirchlet
boundary condition.
"""
struct Value <: AbstractBoundaryConditionClassification end

"""
    NormalFlow

A type specifying the component of a velocity field normal to a boundary.

Thus `NormalFlow` can only be applied to `u` along x, `v` along y, or `w` along z.
For all other cases --- fields located at (Center, Center, Center), or `u`, `v`,
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
struct NormalFlow <: AbstractBoundaryConditionClassification end
