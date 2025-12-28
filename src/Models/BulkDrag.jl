"""
    BulkDrag

Module for implementing quadratic drag boundary conditions for velocity fields.

Provides `BulkDragFunction` for computing surface momentum fluxes using bulk aerodynamic
formulas. The drag function computes a quadratic drag:

```math
τᵘ = - Cᴰ |U| u
```

where `Cᴰ` is the drag coefficient and `|U| = √(u² + v²)` is the horizontal speed.

This is the velocity flux analog of Breeze.jl's `BulkDragFunction`, which computes
momentum flux `Jᵘ = -Cᴰ |U| ρu`.
"""
module BulkDragModule

export BulkDragFunction,
       XDirectionBulkDragFunction,
       YDirectionBulkDragFunction,
       BulkDrag,
       BulkDragBoundaryCondition

using Oceananigans.Architectures: Architectures, on_architecture
using Oceananigans.Grids: AbstractGrid, XDirection, YDirection, Face
using Oceananigans.BoundaryConditions: BoundaryConditions, BoundaryCondition, Flux
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ

using Adapt: Adapt

#####
##### Speed calculations at staggered locations
#####

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2
@inline ϕplusψ²(i, j, k, grid, ϕ, ψ) = @inbounds (ϕ[i, j, k] + ψ)^2

# Speed squared at (Face, Center, Center) - for x-velocity flux
@inline function speed²ᶠᶜᶜ(i, j, k, grid, fields, U∞, V∞)
    u = @inbounds fields.u[i, j, k]
    v² = ℑxyᶠᶜᵃ(i, j, k, grid, ϕplusψ², fields.v, V∞)
    return (u + U∞)^2 + v²
end

# Speed squared at (Center, Face, Center) - for y-velocity flux
@inline function speed²ᶜᶠᶜ(i, j, k, grid, fields, U∞, V∞)
    u² = ℑxyᶜᶠᵃ(i, j, k, grid, ϕplusψ², fields.u, U∞)
    v = @inbounds fields.v[i, j, k]
    return u² + (v + V∞)^2
end

#####
##### BulkDragFunction for velocity fluxes
#####

struct BulkDragFunction{D, C, U}
    direction :: D
    coefficient :: C
    background_velocities :: U
end

"""
    BulkDragFunction(; direction = nothing,
                       coefficient = 1e-3,
                       background_velocities = (0, 0))

Create a bulk drag function for computing surface velocity fluxes using bulk aerodynamic
formulas. The drag function computes a quadratic drag:

```math
τᵘ = - Cᴰ |U + U∞| (u + U∞)
```

where `Cᴰ` is the drag coefficient, `|U + U∞| = √((u + U∞)² + (v + V∞)²)` is the total
horizontal speed including background velocities, and `(U∞, V∞)` are the background velocities.

# Keyword Arguments

- `direction`: The direction of the velocity component (`XDirection()` or `YDirection()`).
               If `nothing`, the direction is inferred from the field location during
               boundary condition regularization.
- `coefficient`: The drag coefficient (default: `1e-3`).
- `background_velocities`: Background velocities as a tuple `(U∞, V∞)` (default: `(0, 0)`).
  These are added to the prognostic velocities when computing both the speed and the drag.
"""
function BulkDragFunction(; direction = nothing,
                            coefficient = 1e-3,
                            background_velocities = (0, 0))
    return BulkDragFunction(direction, coefficient, background_velocities)
end

const XDirectionBulkDragFunction = BulkDragFunction{<:XDirection}
const YDirectionBulkDragFunction = BulkDragFunction{<:YDirection}

Adapt.adapt_structure(to, df::BulkDragFunction) =
    BulkDragFunction(Adapt.adapt(to, df.direction),
                     Adapt.adapt(to, df.coefficient),
                     Adapt.adapt(to, df.background_velocities))

Architectures.on_architecture(to, df::BulkDragFunction) =
    BulkDragFunction(on_architecture(to, df.direction),
                     on_architecture(to, df.coefficient),
                     on_architecture(to, df.background_velocities))

Base.summary(df::BulkDragFunction) = string("BulkDragFunction(direction=", summary(df.direction),
                                            ", coefficient=", df.coefficient, ")")

function Base.show(io::IO, df::BulkDragFunction)
    print(io, summary(df))
end

#####
##### getbc for BulkDragFunction
#####

const XDBDF = XDirectionBulkDragFunction
const YDBDF = YDirectionBulkDragFunction

# Core computation of x-direction drag at a given (i, j, k)
@inline function _x_bulk_drag(i, j, k, grid, fields, Cᴰ, U∞, V∞)
    u = @inbounds fields.u[i, j, k]
    U² = speed²ᶠᶜᶜ(i, j, k, grid, fields, U∞, V∞)
    U = sqrt(U²)
    return - Cᴰ * U * (u + U∞)
end

# Core computation of y-direction drag at a given (i, j, k)
@inline function _y_bulk_drag(i, j, k, grid, fields, Cᴰ, U∞, V∞)
    v = @inbounds fields.v[i, j, k]
    U² = speed²ᶜᶠᶜ(i, j, k, grid, fields, U∞, V∞)
    U = sqrt(U²)
    return - Cᴰ * U * (v + V∞)
end

# Domain boundary getbc (2 indices: i, j) - uses k=1 for bottom boundary
@inline function BoundaryConditions.getbc(df::XDBDF, i::Integer, j::Integer, grid::AbstractGrid, clock, fields)
    U∞, V∞ = df.background_velocities
    return _x_bulk_drag(i, j, 1, grid, fields, df.coefficient, U∞, V∞)
end

@inline function BoundaryConditions.getbc(df::YDBDF, i::Integer, j::Integer, grid::AbstractGrid, clock, fields)
    U∞, V∞ = df.background_velocities
    return _y_bulk_drag(i, j, 1, grid, fields, df.coefficient, U∞, V∞)
end

# Immersed boundary getbc (3 indices: i, j, k) - uses actual k index
@inline function BoundaryConditions.getbc(df::XDBDF, i::Integer, j::Integer, k::Integer, grid::AbstractGrid, clock, fields)
    U∞, V∞ = df.background_velocities
    return _x_bulk_drag(i, j, k, grid, fields, df.coefficient, U∞, V∞)
end

@inline function BoundaryConditions.getbc(df::YDBDF, i::Integer, j::Integer, k::Integer, grid::AbstractGrid, clock, fields)
    U∞, V∞ = df.background_velocities
    return _y_bulk_drag(i, j, k, grid, fields, df.coefficient, U∞, V∞)
end

#####
##### Type alias for FluxBoundaryCondition with BulkDragFunction
#####

const BulkDragBoundaryCondition = BoundaryCondition{<:Flux, <:BulkDragFunction}

#####
##### Regularization: automatically infer direction from field location
#####

"""
    regularize_boundary_condition(df::BulkDragFunction, grid, loc, dim, Side, field_names)

Regularize a `BulkDragFunction` by inferring the direction from the field location if not specified.

The direction is inferred as follows:
- If `loc[1] == Face`, the field is a u-velocity → `XDirection()`
- If `loc[2] == Face`, the field is a v-velocity → `YDirection()`
"""
function BoundaryConditions.regularize_boundary_condition(df::BulkDragFunction{Nothing}, grid, loc, dim, Side, field_names)
    # Infer direction from field location
    if loc[1] isa Face
        direction = XDirection()
    elseif loc[2] isa Face
        direction = YDirection()
    else
        error("Cannot infer BulkDragFunction direction for field at location $loc. " *
              "Please specify direction explicitly using direction=XDirection() or direction=YDirection().")
    end
    return BulkDragFunction(direction, df.coefficient, df.background_velocities)
end

# Already has direction, no regularization needed
regularize_boundary_condition(df::BulkDragFunction, grid, loc, dim, Side, field_names) = df

#####
##### Convenient constructor
#####

"""
    BulkDrag(; direction = nothing,
               coefficient = 1e-3,
               background_velocities = (0, 0))

Create a `FluxBoundaryCondition` for surface velocity drag.

The drag function computes a quadratic drag:

```math
τᵘ = - Cᴰ |U + U∞| (u + U∞)
```

where `Cᴰ` is the drag coefficient, `|U + U∞| = √((u + U∞)² + (v + V∞)²)` is the total
horizontal speed including background velocities, and `(U∞, V∞)` are the background velocities.

See [`BulkDragFunction`](@ref) for details.

# Keyword Arguments

- `direction`: The direction of the velocity component (`XDirection()` or `YDirection()`).
               If `nothing`, the direction is automatically inferred from the field location
               during boundary condition regularization.
- `coefficient`: The drag coefficient (default: `1e-3`).
- `background_velocities`: Background velocities as a tuple `(U∞, V∞)` (default: `(0, 0)`).

# Examples

Create bulk drag boundary conditions for `u` and `v` at the domain bottom.
The direction is automatically inferred from the field location:

```jldoctest
using Oceananigans

drag = BulkDrag(coefficient=1e-3)

u_bcs = FieldBoundaryConditions(bottom=drag)
v_bcs = FieldBoundaryConditions(bottom=drag)

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(; grid, boundary_conditions=(u=u_bcs, v=v_bcs))

# Verify the direction was inferred correctly
model.velocities.u.boundary_conditions.bottom.condition.direction

# output
XDirection()
```

With immersed boundary conditions (the same `BulkDrag` can be used for both
domain boundaries and immersed boundaries):

```jldoctest
using Oceananigans

underlying_grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> -0.5))

drag = BulkDrag(coefficient=1e-3)

# Apply to both domain bottom and immersed boundaries
u_bcs = FieldBoundaryConditions(bottom=drag, immersed=drag)
v_bcs = FieldBoundaryConditions(bottom=drag, immersed=drag)

model = HydrostaticFreeSurfaceModel(; grid, boundary_conditions=(u=u_bcs, v=v_bcs))

# Verify the immersed BC bottom facet has the correct direction
model.velocities.u.boundary_conditions.immersed

# output
ImmersedBoundaryCondition:
├── west: Nothing
├── east: Nothing
├── south: FluxBoundaryCondition: BulkDragFunction(direction=XDirection(), coefficient=0.001)
├── north: FluxBoundaryCondition: BulkDragFunction(direction=XDirection(), coefficient=0.001)
├── bottom: FluxBoundaryCondition: BulkDragFunction(direction=XDirection(), coefficient=0.001)
└── top: FluxBoundaryCondition: BulkDragFunction(direction=XDirection(), coefficient=0.001)
```
"""
function BulkDrag(; kwargs...)
    df = BulkDragFunction(; kwargs...)
    return BoundaryCondition(Flux(), df)
end

end # module
