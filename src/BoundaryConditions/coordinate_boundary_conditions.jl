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

#####
##### Some aliases to make life easier.
#####

const CBC = CoordinateBoundaryConditions

# Here we overload setproperty! and getproperty to permit users to call
# the 'left' and 'right' bcs in the z-direction 'bottom' and 'top'
# and the 'left' and 'right' bcs in the y-direction 'south' and 'north'.
Base.setproperty!(cbc::CBC, side::Symbol, bc) = setbc!(cbc, Val(side), bc)

setbc!(cbc::CBC, ::Val{S}, bc) where S = setfield!(cbc, S, bc)
setbc!(cbc::CBC, ::Val{:bottom}, bc) = setfield!(cbc, :left,  bc)
setbc!(cbc::CBC, ::Val{:top},    bc) = setfield!(cbc, :right, bc)
setbc!(cbc::CBC, ::Val{:south},  bc) = setfield!(cbc, :left,  bc)
setbc!(cbc::CBC, ::Val{:north},  bc) = setfield!(cbc, :right, bc)

Base.getproperty(cbc::CBC, side::Symbol) = getbc(cbc, Val(side))

getbc(cbc::CBC, ::Val{S}) where S = getfield(cbc, S)
getbc(cbc::CBC, ::Val{:bottom}) = getfield(cbc, :left)
getbc(cbc::CBC, ::Val{:top})    = getfield(cbc, :right)
getbc(cbc::CBC, ::Val{:south})  = getfield(cbc, :left)
getbc(cbc::CBC, ::Val{:north})  = getfield(cbc, :right)
