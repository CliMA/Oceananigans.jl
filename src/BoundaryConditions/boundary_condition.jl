import Adapt

"""
    BoundaryCondition{C<:BCType}(condition)

Construct a boundary condition of type `C` with a `condition` that may be given by a
number, an array, or a function with signature:

    condition(i, j, grid, time, iteration, U, Φ, parameters) = # function definition

that returns a number and where `i` and `j` are indices along the boundary.

Boundary condition types include `Periodic`, `Flux`, `Value`, `Gradient`, and `NoPenetration`.
"""
struct BoundaryCondition{C<:BCType, T}
    condition :: T
end

BoundaryCondition(Tbc, c) = BoundaryCondition{Tbc, typeof(c)}(c)

bctype(bc::BoundaryCondition{C}) where C = C

# Adapt boundary condition struct to be GPU friendly and passable to GPU kernels.
Adapt.adapt_structure(to, b::BoundaryCondition{C, A}) where {C<:BCType, A<:AbstractArray} =
    BoundaryCondition(C, Adapt.adapt(to, parent(b.condition)))

#####
##### Some abbreviations to make life easier.
#####

# These type aliases make dispatching on BCs easier (not exported).
const BC   = BoundaryCondition
const FBC  = BoundaryCondition{<:Flux}
const PBC  = BoundaryCondition{<:Periodic}
const NPBC = BoundaryCondition{<:NoPenetration}
const VBC  = BoundaryCondition{<:Value}
const GBC  = BoundaryCondition{<:Gradient}
const NFBC = BoundaryCondition{Flux, Nothing}

# More readable BC constructors for the public API.
     PeriodicBoundaryCondition() = BoundaryCondition(Periodic,      nothing)
NoPenetrationBoundaryCondition() = BoundaryCondition(NoPenetration, nothing)
       NoFluxBoundaryCondition() = BoundaryCondition(Flux,          nothing)

    FluxBoundaryCondition(val) = BoundaryCondition(Flux, val)
   ValueBoundaryCondition(val) = BoundaryCondition(Value, val)
GradientBoundaryCondition(val) = BoundaryCondition(Gradient, val)

# Multiple dispatch on the type of boundary condition
getbc(bc::BC{C, <:Number}, args...)              where C = bc.condition
getbc(bc::BC{C, <:AbstractArray}, i, j, args...) where C = bc.condition[i, j]
getbc(bc::BC{C, <:Function}, args...)            where C = bc.condition(args...)

Base.getindex(bc::BC{C, <:AbstractArray}, inds...) where C = getindex(bc.condition, inds...)
