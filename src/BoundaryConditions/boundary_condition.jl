import Adapt

"""
    BoundaryCondition{C<:BCType}(condition)

Construct a boundary condition of type `C` with a `condition` that may be given by a
number, an array, or a function with signature:

    condition(i, j, grid, clock, state) = # function definition

that returns a number and where `i` and `j` are indices along the boundary.

Boundary condition types include `Periodic`, `Flux`, `Value`, `Gradient`, and `NormalFlow`.
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
const NFBC = BoundaryCondition{<:NormalFlow}
const VBC  = BoundaryCondition{<:Value}
const GBC  = BoundaryCondition{<:Gradient}
const ZFBC = BoundaryCondition{Flux, Nothing} # "zero" flux

# More readable BC constructors for the public API.
    PeriodicBoundaryCondition() = BoundaryCondition(Periodic,   nothing)
      NoFluxBoundaryCondition() = BoundaryCondition(Flux,       nothing)
ImpenetrableBoundaryCondition() = BoundaryCondition(NormalFlow, nothing)

      FluxBoundaryCondition(val) = BoundaryCondition(Flux, val)
     ValueBoundaryCondition(val) = BoundaryCondition(Value, val)
  GradientBoundaryCondition(val) = BoundaryCondition(Gradient, val)
NormalFlowBoundaryCondition(val) = BoundaryCondition(NormalFlow, val)

# Support for various types of boundary conditions
@inline getbc(bc::BC{<:NormalFlow, Nothing}, args...) = 0
@inline getbc(bc::BC{C, <:Number},        i, j, grid, clock, state) where C = bc.condition
@inline getbc(bc::BC{C, <:AbstractArray}, i, j, grid, clock, state) where C = bc.condition[i, j]
@inline getbc(bc::BC{C, <:Function},      i, j, grid, clock, state) where C =
    bc.condition(i, j, grid, clock, state)

@inline Base.getindex(bc::BC{C, <:AbstractArray}, i, j) where C = getindex(bc.condition, i, j)
