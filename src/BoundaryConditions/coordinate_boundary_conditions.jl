"""
    CoordinateBoundaryConditions(left, right)

A set of two `BoundaryCondition`s to be applied along a coordinate x, y, or z.

The `left` boundary condition is applied on the negative or lower side of the coordinate
while the `right` boundary condition is applied on the positive or higher side.
"""
mutable struct CoordinateBoundaryConditions{L, R}
     left :: L
    right :: R
end
