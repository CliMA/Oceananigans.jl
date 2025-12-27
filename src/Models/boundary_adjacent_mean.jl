using Adapt, GPUArraysCore

using Oceananigans.Fields: Center, Face, Field, set!
using Oceananigans.AbstractOperations: Ax, Ay, Az, grid_metric_operation
using Oceananigans.BoundaryConditions: BoundaryCondition, Open

import Oceananigans.BoundaryConditions: update_boundary_condition!, getbc, regularize_boundary_condition

"""
    AverageBoundaryFlux

Computes and stores the area-weighted average of a field `f` on a boundary plane.

The average is computed as `∫f dA / ∫dA` where the integral is over the
boundary-adjacent plane. The total area `∫dA` is pre-computed during
boundary condition regularization.

# Constructor

    AverageBoundaryFlux(side::Symbol)

Create an `AverageBoundaryFlux` for the specified `side` of the domain.
The object is fully initialized during model construction via `regularize_boundary_condition`.

# Arguments
- `side`: One of `:west`, `:east`, `:south`, `:north`, `:bottom`, or `:top`.

# Example

```julia
using Oceananigans
using Oceananigans.Models: AverageBoundaryFlux

# Create boundary conditions with AverageBoundaryFlux
u_bcs = FieldBoundaryConditions(east = OpenBoundaryCondition(AverageBoundaryFlux(:east)))

# The AverageBoundaryFlux is fully initialized during model construction
grid = RectilinearGrid(size=(8, 8, 8), extent=(1, 1, 1))
model = NonhydrostaticModel(; grid, boundary_conditions=(u=u_bcs,))
```
"""
struct AverageBoundaryFlux{S, F, A}
    side :: S   # Val{:east}, etc.
    flux :: F   # Field{Nothing, Nothing, Nothing} for storing result (nothing before regularization)
    area :: A   # Pre-computed total boundary area (nothing before regularization)
end

# User-facing constructor: creates unregularized placeholder
AverageBoundaryFlux(side::Symbol) = AverageBoundaryFlux(Val(side), nothing, nothing)

# For Adapt (GPU transfer)
Adapt.adapt_structure(to, abf::AverageBoundaryFlux) =
    AverageBoundaryFlux(abf.side, adapt(to, abf.flux), abf.area)

Base.show(io::IO, abf::AverageBoundaryFlux) = print(io, summary(abf))

function Base.summary(abf::AverageBoundaryFlux{Val{S}}) where S
    if isnothing(abf.flux)
        return "AverageBoundaryFlux(:$S) (unregularized)"
    else
        return "AverageBoundaryFlux(:$S): $(@allowscalar first(abf.flux))"
    end
end

# For boundary condition value access
@inline getbc(abf::AverageBoundaryFlux, args...) = @allowscalar first(abf.flux)

#####
##### Regularization: build full object during model construction
#####

is_regularized(abf::AverageBoundaryFlux) = !isnothing(abf.flux)

function regularize_boundary_condition(abf::AverageBoundaryFlux,
                                       grid, loc, dim, Side, field_names)

    # If already regularized, return as-is
    is_regularized(abf) && return abf

    flux = Field{Nothing, Nothing, Nothing}(grid)

    # Pre-compute total boundary area
    i, j, k = boundary_view_indices(abf.side, grid)

    # Compute the boundary-normal area
    An = boundary_area_metric(abf.side)
    An_operation = grid_metric_operation(loc, An, grid)
    An_field = Field(An_operation, indices=(i, j, k))
    area = sum(An_field)

    return AverageBoundaryFlux(abf.side, flux, area)
end

#####
##### Boundary view indices
#####

# Returns (i, j, k) indices for view() to extract the boundary-adjacent plane
@inline boundary_view_indices(::Val{:west}, grid)   = (1:1, :, :)
@inline boundary_view_indices(::Val{:east}, grid)   = (size(grid, 1):size(grid, 1), :, :)
@inline boundary_view_indices(::Val{:south}, grid)  = (:, 1:1, :)
@inline boundary_view_indices(::Val{:north}, grid)  = (:, size(grid, 2):size(grid, 2), :)
@inline boundary_view_indices(::Val{:bottom}, grid) = (:, :, 1:1)
@inline boundary_view_indices(::Val{:top}, grid)    = (:, :, size(grid, 3):size(grid, 3))

#####
##### Boundary-normal area metric
#####

@inline boundary_area_metric(::Union{Val{:west}, Val{:east}})   = Ax
@inline boundary_area_metric(::Union{Val{:south}, Val{:north}}) = Ay
@inline boundary_area_metric(::Union{Val{:bottom}, Val{:top}})  = Az

#####
##### Compute the boundary average
#####

const OpenBC_ABF = BoundaryCondition{<:Open, <:AverageBoundaryFlux}

@inline function update_boundary_condition!(bc::OpenBC_ABF, val_side, u, model)
    abf = bc.condition
    grid = u.grid

    # Get the boundary slice of u
    i, j, k = boundary_view_indices(abf.side, grid)
    u_boundary = view(u, i, j, k)

    # Get the area metric
    An = boundary_area_metric(abf.side)

    # Compute area-weighted sum: ∫u dA
    sum!(abf.flux, u_boundary * An)

    # Divide by pre-computed area to get average
    @allowscalar abf.flux[1, 1, 1] /= abf.area

    return nothing
end
