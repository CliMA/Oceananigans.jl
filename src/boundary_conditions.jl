#
# Boundaries and Boundary Conditions
#

const coordinates = (:x, :y, :z)
const dims = length(coordinates)
const solution_fields = (:u, :v, :w, :T, :S)
const nsolution = length(solution_fields)

abstract type BCType end
struct Default <: BCType end
abstract type NonDefault <: BCType end
struct Flux <: NonDefault end
struct Gradient <: NonDefault end
struct Value <: NonDefault end

"""
    BoundaryCondition(condition)


"""
struct BoundaryCondition{C<:BCType, T}
    condition::T
end

# Implements a sugary callable BC.
(bc::BoundaryCondition{<:BCType, <:Function})(args...) = bc.condition(args...)

# Constructors
BoundaryCondition(Tbc, c) = BoundaryCondition{Tbc, typeof(c)}(c)

"""
    TimeVaryingBoundaryCondition(T, f)

Construct a boundary condition f(t) as a function of time only.
"""
function TimeVaryingBoundaryCondition(T::BCType, f)
  @inline condition(t, args...) = f(t)
  BoundaryCondition{T, Function}(condition)
end

DefaultBC() = BoundaryCondition{Default, Nothing}(nothing)

"""
    CoordinateBoundaryConditions(c)

Construct `CoordinateBoundaryCondition` to be applied along coordinate `c`, where
`c` is `:x`, `:y`, or `:z`. A CoordinateBoundaryCondition has two fields
`left` and `right` that store boundary conditions on the 'left' (negative side)
and 'right' (positive side) of a given coordinate.
"""
mutable struct CoordinateBoundaryConditions <: FieldVector{2, BoundaryCondition}
  left::BoundaryCondition
  right::BoundaryCondition
end

CoordinateBoundaryConditions() = CoordinateBoundaryConditions(DefaultBC(), DefaultBC())

"""
    FieldBoundaryConditions()

Construct `FieldBoundaryConditions` for a field.
A FieldBoundaryCondition has `CoordinateBoundaryConditions` in
`x`, `y`, and `z`.
"""
struct FieldBoundaryConditions <: FieldVector{dims, CoordinateBoundaryConditions}
  x::CoordinateBoundaryConditions
  y::CoordinateBoundaryConditions
  z::CoordinateBoundaryConditions
end

"""
    BoundaryConditions()

Construct a boundary condition type full of default
`FieldBoundaryConditions` for u, v, w, T, S.
"""
struct ModelBoundaryConditions <: FieldVector{nsolution, FieldBoundaryConditions}
  u::FieldBoundaryConditions
  v::FieldBoundaryConditions
  w::FieldBoundaryConditions
  T::FieldBoundaryConditions
  S::FieldBoundaryConditions
end

FieldBoundaryConditions() = FieldBoundaryConditions(CoordinateBoundaryConditions(),
                                                    CoordinateBoundaryConditions(),
                                                    CoordinateBoundaryConditions())

function ModelBoundaryConditions()
  bcs = (FieldBoundaryConditions() for i = 1:length(solution_fields))
  return ModelBoundaryConditions(bcs...)
end

#
# User API
#
# Note:
#
# The syntax model.boundary_conditions.u.x.left = bc works, out of the box.
# How can we make it easier for users to set boundary conditions?

const BC = BoundaryCondition
const FBCs = FieldBoundaryConditions
