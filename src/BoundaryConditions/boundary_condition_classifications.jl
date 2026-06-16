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
    struct Value{MS} <: AbstractBoundaryConditionClassification

A classification specifying a boundary condition on the value of a field. Also called a Dirichlet
boundary condition.

The optional `scheme` (type parameter `MS`) selects a "matching scheme" that determines how the
boundary value is computed from the prescribed exterior value and the interior state. With the
default `scheme = nothing` the prescribed value is imposed directly. A scheme such as
[`PerturbationAdvection`](@ref) instead radiates a Center-located field (e.g. a tracer) out of
the domain, nudging it towards the prescribed exterior value.
"""
struct Value{MS} <: AbstractBoundaryConditionClassification
    scheme :: MS
end

Value() = Value(nothing)

(value::Value)() = value

Adapt.adapt_structure(to, value::Value) = Value(adapt(to, value.scheme))

"""
    struct Mixed <: AbstractBoundaryConditionClassification

A classification specifying a boundary condition that represents a linear combination of
the field's gradient and value. Also called a Robin boundary condition.
"""
struct Mixed <: AbstractBoundaryConditionClassification end

"""
    struct NormalFlow{MS} <: AbstractBoundaryConditionClassification

A classification that specifies the boundary-normal flow on a `Face`-located boundary, and
thereby the halo region of the field directly.

`NormalFlow` is used to specify the component of a velocity field normal to a boundary; because
that component lives _on_ the boundary (at a `Face`), the classification sets the field value on
the boundary itself. It can also be used to describe nested or linked simulation domains.

The optional `scheme` (type parameter `MS`) selects a "matching scheme". With the default
`scheme = nothing` the prescribed value is imposed directly (an imposed normal velocity). A
scheme such as [`PerturbationAdvection`](@ref) instead radiates the boundary-normal velocity out
of the domain.
"""
struct NormalFlow{MS} <: AbstractBoundaryConditionClassification
    scheme :: MS
end

NormalFlow() = NormalFlow(nothing)

(normal_flow::NormalFlow)() = normal_flow

Adapt.adapt_structure(to, normal_flow::NormalFlow) = NormalFlow(adapt(to, normal_flow.scheme))

"""
    struct MultiRegionCommunication <: AbstractBoundaryConditionClassification

A classification specifying a shared memory communicating boundary condition
"""
struct MultiRegionCommunication <: AbstractBoundaryConditionClassification end

"""
    struct DistributedCommunication <: AbstractBoundaryConditionClassification

A classification specifying a distributed memory communicating boundary condition
"""
struct DistributedCommunication <: AbstractBoundaryConditionClassification end

"""
    AbstractPivot

An abstract type representing pivot locations for Zipper boundary conditions.
"""
abstract type AbstractPivot end

"""
    struct UPivot <: AbstractPivot

The type representing a U-point pivot for Zipper boundary conditions.

See [`TripolarGrid`](@ref) for examples.
"""
struct UPivot <: AbstractPivot end

"""
    struct FPivot <: AbstractPivot

The type representing a F-point pivot for Zipper boundary conditions.

See [`TripolarGrid`](@ref) for examples.
"""
struct FPivot <: AbstractPivot end


"""
    Zipper{P} <: AbstractBoundaryConditionClassification

A classification specifying a Zipper boundary condition where one boundary is folded onto itself.
The points where the zipper starts and ends act as "pivots", and the grid cell location of these pivot points
is encoded in the type parameter `P`.
`P` can be set to either:
- `UPivot`: pivots on (Face, Center)
- `FPivot`: pivots on (Face, Face)

See [`TripolarGrid`](@ref) for examples.
"""
struct Zipper{P <: AbstractPivot} <: AbstractBoundaryConditionClassification end

"""
    pivot_type(zbc::Zipper)

Returns the pivot type of the Zipper boundary condition `zbc`.
"""
pivot_type(::Zipper{T}) where T = T
