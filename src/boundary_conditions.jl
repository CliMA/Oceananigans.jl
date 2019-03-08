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

# TODO: 
#   * figure out how to use functors for boundary conditions.
#   * figure out how to declare the type of the boundary condition in struct definition.

#=
abstract type BCType end
struct Flux <: BCType end
struct Value <: BCType end
struct Default <: BCType end

mutable struct BoundaryCondition{TBC, C, B} 
    call::Function
end

struct CoordinateBoundaryConditions{TL, TR}
  left::TL
  right::TR
end

struct FieldBoundaryConditions{TX, TY, TZ}
  x::TX
  y::TY
  z::TZ
end

struct FieldBoundaryConditions <: FieldVector{

struct BoundaryConditions{TU, TV <: FieldVector{

=#

abstract type BoundaryCondition{C, B} end
const BC = BoundaryCondition # alias

struct DefaultBC{C, B} <: BoundaryCondition{C, B} end

struct FluxBC{C, B} <: BoundaryCondition{C, B}
  flux::Function
end

struct ValueBC{C, B} <: BoundaryCondition{C, B}
  value::Function
end

"""
    CoordinateBoundaryConditions(c)

Construct `CoordinateBoundaryCondition` to be applied along coordinate `c`, where
`c` is `:x`, `:y`, or `:z`. A CoordinateBoundaryCondition has two fields
`left` and `right` that store boundary conditions on the 'left' (negative side) 
and 'right' (positive side) of a given coordinate.
"""
mutable struct CoordinateBoundaryConditions{C}
  left::BoundaryCondition{C, :left} 
  right::BoundaryCondition{C, :right} 
end

CoordinateBoundaryConditions(c) = CoordinateBoundaryConditions{c}(DefaultBC{c, :left}(), DefaultBC{c, :right}())

"""
    FieldBoundaryConditions()

Construct `FieldBoundaryConditions` for a field.
A FieldBoundaryCondition has fields `x`, `y`, and `z`
cooresponding to `CoordinateBoundaryConditions`in those respective directions.
"""
struct FieldBoundaryConditions
  x::CoordinateBoundaryConditions{:x}
  y::CoordinateBoundaryConditions{:y}
  z::CoordinateBoundaryConditions{:z}
end

FieldBoundaryConditions() = FieldBoundaryConditions(CoordinateBoundaryConditions(:x),
                                                    CoordinateBoundaryConditions(:y),
                                                    CoordinateBoundaryConditions(:z))

"""
    BoundaryConditions()

Construct a boundary condition type full of default FieldBoundaryConditions for u, v, w, T, S.
"""
struct BoundaryConditions <: FieldVector{5, FieldBoundaryConditions}
  u::FieldBoundaryConditions
  v::FieldBoundaryConditions
  w::FieldBoundaryConditions
  T::FieldBoundaryConditions
  S::FieldBoundaryConditions
end

function BoundaryConditions()
  return BoundaryConditions(
    FieldBoundaryConditions(),
    FieldBoundaryConditions(),
    FieldBoundaryConditions(),
    FieldBoundaryConditions(),
    FieldBoundaryConditions()
)
end

# A few sugary things

# allows the syntax: model.boundary_conditions.u = (bc1, bc2) or bcs.u = bc1
Base.setproperty!(bcs::BoundaryConditions, fld, bc::Union{BC, NTuple{2, BC}}) = add_bc!(bcs, fld, bc)

"""
    add_bcs!(model, u=...)

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
