"""
    AbstractBoundaryConditionClassification

Abstract supertype for boundary condition types.
"""
abstract type AbstractBoundaryConditionClassification end

"""
    struct Periodic <: AbstractBoundaryConditionClassification

A classification specifying a periodic boundary condition.

A condition may not be specified with a `Periodic` boundary condition.
"""
struct Periodic <: AbstractBoundaryConditionClassification end

"""
    struct Flux <: AbstractBoundaryConditionClassification

A classification specifying a boundary condition on the flux of a field.

The sign convention is such that a positive flux represents the flux of a quantity in the
positive direction. For example, a positive vertical flux implies a quantity is fluxed
upwards, in the ``+z`` direction.

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
    struct Gradient <: AbstractBoundaryConditionClassification

A classification specifying a boundary condition on the derivative or gradient of a field. Also
called a Neumann boundary condition.
"""
struct Gradient <: AbstractBoundaryConditionClassification end

"""
    struct Value <: AbstractBoundaryConditionClassification

A classification specifying a boundary condition on the value of a field. Also called a Dirchlet
boundary condition.
"""
struct Value <: AbstractBoundaryConditionClassification end

"""
    struct Open <: AbstractBoundaryConditionClassification

A classification that specifies the halo regions of a field directly.

For fields located at Faces, Open also specifies field value _on_ the boundary.

Open boundary conditions are used to specify the component of a velocity field normal to a boundary
and can also be used to describe nested or linked simulation domains.
"""
struct Open <: AbstractBoundaryConditionClassification end

"""
    struct Communication <: AbstractBoundaryConditionClassification

A classification specifying a communicating boundary condition
"""
struct Communication <: AbstractBoundaryConditionClassification end