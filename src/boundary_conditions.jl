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
    BoundaryCondition(BCType, condition)

Construct a boundary condition of `BCType` with `condition`,
where `BCType` is `Flux` or `Gradient`. `condition` may be a
number, array, or a function with signature:

    condition(t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S, iteration, i, j) = # function definition

`i` and `j` are indices along the boundary.
"""
struct BoundaryCondition{C<:BCType, T}
    condition::T
end

# Constructors
BoundaryCondition(Tbc, c) = BoundaryCondition{Tbc, typeof(c)}(c)

Adapt.adapt_structure(to, b::BoundaryCondition{C, A}) where {C<:BCType, A<:AbstractArray} = BoundaryCondition(C, Adapt.adapt(to, parent(b.condition)))

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
mutable struct CoordinateBoundaryConditions
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

#
# Physics goes here.
#

#=
Currently we support flux and gradient boundary conditions
at the top and bottom of the domain.

Notes:

- The boundary condition on a z-boundary is a callable object with arguments

      (t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S, iteration, i, j),

  where i and j are the x and y indices, respectively. No other function signature will work.
  We do not have abstractions that generalize to non-uniform grids.

- We assume that the boundary tendency has been previously calculated for
  a 'no-flux' boundary condition.

  This means that boudnary conditions take the form of
  an addition/subtraction to the tendency associated with a flux at point (A, A, I) below the bottom cell.
  This paradigm holds as long as consider boundary conditions on (A, A, C) variables only, where A is
  "any" of C or I.

 - We use the physics-based convention that

        flux = -κ * gradient,

    and that

        tendency = ∂ϕ/∂t = Gϕ = - ∇ ⋅ flux

=#

# Do nothing in default case. These functions are called in cases where one of the
# z-boundaries is set, but not the other.
@inline apply_z_top_bc!(args...) = nothing
@inline apply_z_bottom_bc!(args...) = nothing

# These functions compute vertical fluxes for (A, A, C) quantities. They are not currently used.
@inline ∇κ∇ϕ_t(κ, ϕt, ϕt₋₁, flux, ΔzC, ΔzF) = (      -flux        - κ*(ϕt - ϕt₋₁)/ΔzC ) / ΔzF
@inline ∇κ∇ϕ_b(κ, ϕb, ϕb₊₁, flux, ΔzC, ΔzF) = ( κ*(ϕb₊₁ - ϕb)/ΔzC +       flux        ) / ΔzF

getbc(bc::BC{C, <:Function},      t, grid, u, v, w, T, S, iteration, i, j) where C = bc.condition(t, grid, u, v, w, T, S, iteration, i, j)
getbc(bc::BC{C, <:Number},        t, grid, u, v, w, T, S, iteration, i, j) where C = bc.condition
getbc(bc::BC{C, <:AbstractArray}, t, grid, u, v, w, T, S, iteration, i, j) where C = bc.condition[i, j]

"Add flux divergence to ∂ϕ/∂t associated with a top boundary condition on ϕ."
@inline apply_z_top_bc!(top_flux::BC{<:Flux}, ϕ, Gϕ, κ, t, grid, u, v, w, T, S, iteration, i, j) =
    Gϕ[i, j, 1] -= getbc(top_flux, t, grid, u, v, w, T, S, iteration, i, j) / grid.Δz

@inline apply_z_top_bc!(top_gradient::BC{<:Gradient}, ϕ, Gϕ, κ, t, grid, u, v, w, T, S, iteration, i, j) =
    Gϕ[i, j, 1] += κ * getbc(top_gradient, t, grid, u, v, w, T, S, iteration, i, j) / grid.Δz

"Add flux divergence to ∂ϕ/∂t associated with a bottom boundary condition on ϕ."
@inline apply_z_bottom_bc!(bottom_flux::BC{<:Flux}, ϕ, Gϕ, κ, t, grid, u, v, w, T, S, iteration, i, j) =
    Gϕ[i, j, grid.Nz] += getbc(bottom_flux, t, grid, u, v, w, T, S, iteration, i, j) / grid.Δz

@inline apply_z_bottom_bc!(bottom_gradient::BC{<:Gradient}, ϕ, Gϕ, κ, t, grid, u, v, w, T, S, iteration, i, j) =
    Gϕ[i, j, grid.Nz] -= κ * getbc(bottom_gradient, t, grid, u, v, w, T, S, iteration, i, j) / grid.Δz
