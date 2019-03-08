#=
struct BoundaryConditions
    x_bc::Symbol
    y_bc::Symbol
    top_bc::Symbol
    bottom_bc::Symbol

    function BoundaryConditions(x_bc, y_bc, top_bc, bottom_bc)
        @assert x_bc == :periodic && y_bc == :periodic "Only periodic horizontal boundary conditions are currently supported."
        @assert top_bc == :rigid_lid "Only rigid lid is currently supported at the top."
        @assert bottom_bc in [:no_slip, :free_slip] "Bottom boundary condition must be :no_slip or :free_slip."
        new(x_bc, y_bc, top_bc, bottom_bc)
    end
end
=#


#
# Boundaries and Boundary Conditions
#

const coordinates = (:x, :y, :z)
const dims = length(coordinates)
const solution_fields = (:u, :v, :w, :T, :S)

abstract type Condition end
struct Default <: Condition end
struct Flux <: Condition end
struct Value <: Condition end

struct BoundaryCondition{C}
    calc::Function
end

(bc::BoundaryCondition)(args...) = bc.calc(args...)

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
struct BoundaryConditions <: FieldVector{5, FieldBoundaryConditions}
  u::FieldBoundaryConditions
  v::FieldBoundaryConditions
  w::FieldBoundaryConditions
  T::FieldBoundaryConditions
  S::FieldBoundaryConditions
end

FieldBoundaryConditions() = FieldBoundaryConditions(CoordinateBoundaryConditions(),
                                                    CoordinateBoundaryConditions(),
                                                    CoordinateBoundaryConditions())

function BoundaryConditions()
  bcs = (FieldBoundaryConditions() for i = 1:length(solution_fields)))
  return BoundaryConditions(bcs...)
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

#=
"""
    add_bcs!(boundary_conditions, u=...)

Add `bc` as a boundary condition of `fld`.
"""
function add_bcs!(bcs; kwargs...)
  for (k, v) in kwargs
    add_bc!(bcs, k, v)
  end
end

"""
    add_bc!(boundary_conditions, fld, bc)

Add `bc` as a boundary condition of `fld`.
"""
function add_bc!(boundary_conditions, fld, bc::BoundaryCondition{C, B}) where {C, B}
  field_bcs = getproperty(boundary_conditions, fld)
  setproperty!(getproperty(field_bcs, C), B, bc)
end

function add_bc!(bcs, fld, bctuple::NTuple{2, BC})
  add_bc!(bcs, fld, bctuple[1])
  add_bc!(bcs, fld, bctuple[2])
  return nothing
end
=#
