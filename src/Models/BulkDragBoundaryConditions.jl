"""
    BulkDragBoundaryConditions

Module for implementing drag boundary conditions for velocity fields on
all domain boundaries (west, east, south, north, bottom, top) and immersed boundaries.

Provides `BulkDragFunction` for computing momentum fluxes using bulk aerodynamic
formulas. The drag function computes a drag on the tangential velocity. For example for
a quadratic drag:

```math
τᵘ = - Cᴰ |U| u
```

The boundary-normal velocity component is zero due to the no-penetration condition.
"""
module BulkDragBoundaryConditions

export BulkDragFunction,
       XDirectionBulkDragFunction,
       YDirectionBulkDragFunction,
       ZDirectionBulkDragFunction,
       BulkDrag,
       BulkDragBoundaryCondition,
       LinearFormulation,
       QuadraticFormulation

using Oceananigans.Architectures: Architectures, on_architecture
using Oceananigans.Grids: AbstractGrid, XDirection, YDirection, ZDirection, Face
using Oceananigans.BoundaryConditions: BoundaryConditions, BoundaryCondition, Flux,
                                       LeftBoundary, RightBoundary
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxzᶠᵃᶜ, ℑyzᵃᶠᶜ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶠ

using Adapt: Adapt

#####
##### Drag formulations
#####

"""
    LinearFormulation()

Linear formulation for [`BulkDrag`](@ref): `τ = -Cᴰ * u`,
where `Cᴰ` is the drag coefficient and has units of velocity.
Also known as Rayleigh friction.
"""
struct LinearFormulation end

"""
    QuadraticFormulation()

Quadratic drag formulation for [`BulkDrag`](@ref): `τ = -Cᴰ * |U| * u`,
where `Cᴰ` is the non-dimensional drag coefficient.
The drag is proportional to velocity times speed (quadratic in velocity magnitude).
This is the standard bulk aerodynamic formula.
"""
struct QuadraticFormulation end

Base.summary(::LinearFormulation) = "LinearFormulation()"
Base.summary(::QuadraticFormulation) = "QuadraticFormulation()"

#####
##### Speed calculations at staggered locations
#####

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2
@inline ϕplusψ²(i, j, k, grid, ϕ, ψ) = @inbounds (ϕ[i, j, k] + ψ)^2

# Speed² at (Face, Center, Center) - for x-velocity (u) drag
# Uses all three velocity components; boundary-normal component is zero due to masking
@inline function speed²ᶠᶜᶜ(i, j, k, grid, fields, U∞, V∞, W∞)
    u = @inbounds fields.u[i, j, k]
    v² = ℑxyᶠᶜᵃ(i, j, k, grid, ϕplusψ², fields.v, V∞)
    w² = ℑxzᶠᵃᶜ(i, j, k, grid, ϕplusψ², fields.w, W∞)
    return (u + U∞)^2 + v² + w²
end

# Speed² at (Center, Face, Center) - for y-velocity (v) drag
@inline function speed²ᶜᶠᶜ(i, j, k, grid, fields, U∞, V∞, W∞)
    u² = ℑxyᶜᶠᵃ(i, j, k, grid, ϕplusψ², fields.u, U∞)
    v = @inbounds fields.v[i, j, k]
    w² = ℑyzᵃᶠᶜ(i, j, k, grid, ϕplusψ², fields.w, W∞)
    return u² + (v + V∞)^2 + w²
end

# Speed² at (Center, Center, Face) - for z-velocity (w) drag
@inline function speed²ᶜᶜᶠ(i, j, k, grid, fields, U∞, V∞, W∞)
    u² = ℑxzᶜᵃᶠ(i, j, k, grid, ϕplusψ², fields.u, U∞)
    v² = ℑyzᵃᶜᶠ(i, j, k, grid, ϕplusψ², fields.v, V∞)
    w = @inbounds fields.w[i, j, k]
    return u² + v² + (w + W∞)^2
end

#####
##### BulkDragFunction for velocity fluxes
#####

struct BulkDragFunction{D, S, M, F, C, U}
    direction :: D              # XDirection, YDirection, or ZDirection
    side :: S                   # LeftBoundary(), RightBoundary(), or nothing
    dim :: M                    # Boundary dimension (1, 2, or 3) or nothing
    formulation :: F            # LinearFormulation() or QuadraticFormulation()
    coefficient :: C
    background_velocities :: U  # (U∞, V∞, W∞)
end

function BulkDragFunction(formulation=QuadraticFormulation(); coefficient,
                          direction = nothing,
                          background_velocities = (0, 0, 0))
    # side and dim are set during regularization
    return BulkDragFunction(direction, nothing, nothing, formulation, coefficient, background_velocities)
end

const XDirectionBulkDragFunction{S} = BulkDragFunction{<:XDirection, S} where S
const YDirectionBulkDragFunction{S} = BulkDragFunction{<:YDirection, S} where S
const ZDirectionBulkDragFunction{S} = BulkDragFunction{<:ZDirection, S} where S

Adapt.adapt_structure(to, df::BulkDragFunction) =
    BulkDragFunction(Adapt.adapt(to, df.direction),
                     Adapt.adapt(to, df.side),
                     Adapt.adapt(to, df.dim),
                     Adapt.adapt(to, df.formulation),
                     Adapt.adapt(to, df.coefficient),
                     Adapt.adapt(to, df.background_velocities))

Architectures.on_architecture(to, df::BulkDragFunction) =
    BulkDragFunction(on_architecture(to, df.direction),
                     on_architecture(to, df.side),
                     on_architecture(to, df.dim),
                     on_architecture(to, df.formulation),
                     on_architecture(to, df.coefficient),
                     on_architecture(to, df.background_velocities))

Base.summary(df::BulkDragFunction) = string("BulkDragFunction(", summary(df.formulation),
                                            ", ", summary(df.direction),
                                            ", Cᴰ=", df.coefficient, ")")

function Base.show(io::IO, df::BulkDragFunction)
    print(io, summary(df))
end

#####
##### getbc for BulkDragFunction
#####

const XDBDF = XDirectionBulkDragFunction
const YDBDF = YDirectionBulkDragFunction
const ZDBDF = ZDirectionBulkDragFunction

#####
##### Core drag computations dispatched on formulation
#####

# Quadratic drag: τ = -Cᴰ |U| u
@inline function _x_bulk_drag(i, j, k, grid, ::QuadraticFormulation, fields, Cᴰ, U∞, V∞, W∞)
    u = @inbounds fields.u[i, j, k]
    U² = speed²ᶠᶜᶜ(i, j, k, grid, fields, U∞, V∞, W∞)
    U = sqrt(U²)
    return - Cᴰ * U * (u + U∞)
end

@inline function _y_bulk_drag(i, j, k, grid, ::QuadraticFormulation, fields, Cᴰ, U∞, V∞, W∞)
    v = @inbounds fields.v[i, j, k]
    U² = speed²ᶜᶠᶜ(i, j, k, grid, fields, U∞, V∞, W∞)
    U = sqrt(U²)
    return - Cᴰ * U * (v + V∞)
end

@inline function _z_bulk_drag(i, j, k, grid, ::QuadraticFormulation, fields, Cᴰ, U∞, V∞, W∞)
    w = @inbounds fields.w[i, j, k]
    U² = speed²ᶜᶜᶠ(i, j, k, grid, fields, U∞, V∞, W∞)
    U = sqrt(U²)
    return - Cᴰ * U * (w + W∞)
end

# Linear drag (Rayleigh friction): τ = -Cᴰ u
@inline function _x_bulk_drag(i, j, k, grid, ::LinearFormulation, fields, Cᴰ, U∞, V∞, W∞)
    u = @inbounds fields.u[i, j, k]
    return - Cᴰ * (u + U∞)
end

@inline function _y_bulk_drag(i, j, k, grid, ::LinearFormulation, fields, Cᴰ, U∞, V∞, W∞)
    v = @inbounds fields.v[i, j, k]
    return - Cᴰ * (v + V∞)
end

@inline function _z_bulk_drag(i, j, k, grid, ::LinearFormulation, fields, Cᴰ, U∞, V∞, W∞)
    w = @inbounds fields.w[i, j, k]
    return - Cᴰ * (w + W∞)
end

#####
##### Helper for boundary index
#####

@inline boundary_index(::LeftBoundary,  N) = 1
@inline boundary_index(::RightBoundary, N) = N

#####
##### Domain boundary getbc methods (2-index signatures)
#####

# Type aliases for dimension dispatch
const XNormalBulkDragFunction{D} = BulkDragFunction{D, <:Any, Val{1}} where D  # dim=1: west/east
const YNormalBulkDragFunction{D} = BulkDragFunction{D, <:Any, Val{2}} where D  # dim=2: south/north
const ZNormalBulkDragFunction{D} = BulkDragFunction{D, <:Any, Val{3}} where D  # dim=3: bottom/top

# z-normal boundaries (bottom/top): getbc(df, i, j, grid, ...)
# Applies to u-velocity (XDirection) and v-velocity (YDirection)
@inline function BoundaryConditions.getbc(df::ZNormalBulkDragFunction{<:XDirection}, i::Integer, j::Integer,
                                          grid::AbstractGrid, clock, fields, args...)
    U∞, V∞, W∞ = df.background_velocities
    k = boundary_index(df.side, grid.Nz)
    return _x_bulk_drag(i, j, k, grid, df.formulation, fields, df.coefficient, U∞, V∞, W∞)
end

@inline function BoundaryConditions.getbc(df::ZNormalBulkDragFunction{<:YDirection}, i::Integer, j::Integer,
                                          grid::AbstractGrid, clock, fields, args...)
    U∞, V∞, W∞ = df.background_velocities
    k = boundary_index(df.side, grid.Nz)
    return _y_bulk_drag(i, j, k, grid, df.formulation, fields, df.coefficient, U∞, V∞, W∞)
end

# x-normal boundaries (west/east): getbc(df, j, k, grid, ...)
# Applies to v-velocity (YDirection) and w-velocity (ZDirection)
@inline function BoundaryConditions.getbc(df::XNormalBulkDragFunction{<:YDirection}, j::Integer, k::Integer,
                                          grid::AbstractGrid, clock, fields, args...)
    U∞, V∞, W∞ = df.background_velocities
    i = boundary_index(df.side, grid.Nx)
    return _y_bulk_drag(i, j, k, grid, df.formulation, fields, df.coefficient, U∞, V∞, W∞)
end

@inline function BoundaryConditions.getbc(df::XNormalBulkDragFunction{<:ZDirection}, j::Integer, k::Integer,
                                          grid::AbstractGrid, clock, fields, args...)
    U∞, V∞, W∞ = df.background_velocities
    i = boundary_index(df.side, grid.Nx)
    return _z_bulk_drag(i, j, k, grid, df.formulation, fields, df.coefficient, U∞, V∞, W∞)
end

# y-normal boundaries (south/north): getbc(df, i, k, grid, ...)
# Applies to u-velocity (XDirection) and w-velocity (ZDirection)
@inline function BoundaryConditions.getbc(df::YNormalBulkDragFunction{<:XDirection}, i::Integer, k::Integer,
                                          grid::AbstractGrid, clock, fields, args...)
    U∞, V∞, W∞ = df.background_velocities
    j = boundary_index(df.side, grid.Ny)
    return _x_bulk_drag(i, j, k, grid, df.formulation, fields, df.coefficient, U∞, V∞, W∞)
end

@inline function BoundaryConditions.getbc(df::YNormalBulkDragFunction{<:ZDirection}, i::Integer, k::Integer,
                                          grid::AbstractGrid, clock, fields, args...)
    U∞, V∞, W∞ = df.background_velocities
    j = boundary_index(df.side, grid.Ny)
    return _z_bulk_drag(i, j, k, grid, df.formulation, fields, df.coefficient, U∞, V∞, W∞)
end

#####
##### Immersed boundary getbc methods (3-index signatures)
#####

# Immersed boundaries always use (i, j, k) explicitly
@inline function BoundaryConditions.getbc(df::XDBDF, i::Integer, j::Integer, k::Integer,
                                          grid::AbstractGrid, clock, fields, args...)
    U∞, V∞, W∞ = df.background_velocities
    return _x_bulk_drag(i, j, k, grid, df.formulation, fields, df.coefficient, U∞, V∞, W∞)
end

@inline function BoundaryConditions.getbc(df::YDBDF, i::Integer, j::Integer, k::Integer,
                                          grid::AbstractGrid, clock, fields, args...)
    U∞, V∞, W∞ = df.background_velocities
    return _y_bulk_drag(i, j, k, grid, df.formulation, fields, df.coefficient, U∞, V∞, W∞)
end

@inline function BoundaryConditions.getbc(df::ZDBDF, i::Integer, j::Integer, k::Integer,
                                          grid::AbstractGrid, clock, fields, args...)
    U∞, V∞, W∞ = df.background_velocities
    return _z_bulk_drag(i, j, k, grid, df.formulation, fields, df.coefficient, U∞, V∞, W∞)
end

#####
##### Type alias for FluxBoundaryCondition with BulkDragFunction
#####

const BulkDragBoundaryCondition = BoundaryCondition{<:Flux, <:BulkDragFunction}

#####
##### Regularization: infer direction from field location and set Side
#####

"""
    regularize_boundary_condition(df::BulkDragFunction, grid, loc, dim, Side, field_names)

Regularize a `BulkDragFunction` by:
1. Inferring the direction from the field location if not specified
2. Setting the boundary side and dimension from the regularization context

The direction is inferred as follows:
- If `loc[1] == Face`, the field is a u-velocity → `XDirection()`
- If `loc[2] == Face`, the field is a v-velocity → `YDirection()`
- If `loc[3] == Face`, the field is a w-velocity → `ZDirection()`

The dimension `dim` indicates which axis the boundary is normal to:
- `dim=1`: x-normal boundary (west/east)
- `dim=2`: y-normal boundary (south/north)
- `dim=3`: z-normal boundary (bottom/top)
"""
function BoundaryConditions.regularize_boundary_condition(df::BulkDragFunction{Nothing}, grid, loc, dim, Side, field_names)
    # Infer direction from field location
    if loc[1] isa Face
        direction = XDirection()
    elseif loc[2] isa Face
        direction = YDirection()
    elseif loc[3] isa Face
        direction = ZDirection()
    else
        error("Cannot infer BulkDragFunction direction for field at location $loc. " *
              "Please specify direction explicitly.")
    end
    # Side() instantiates the Side type (e.g., LeftBoundary → LeftBoundary())
    # Val{dim} is used for type dispatch in getbc methods
    return BulkDragFunction(direction, Side(), Val{dim}(), df.formulation, df.coefficient, df.background_velocities)
end

# Direction already specified, just set the Side and dim
function BoundaryConditions.regularize_boundary_condition(df::BulkDragFunction, grid, loc, dim, Side, field_names)
    return BulkDragFunction(df.direction, Side(), Val{dim}(), df.formulation, df.coefficient, df.background_velocities)
end

#####
##### Convenient constructor
#####

"""
    BulkDrag(formulation=QuadraticFormulation(); coefficient, background_velocities=(0, 0, 0))

Create a `FluxBoundaryCondition` for velocity drag on any boundary.

# Positional Arguments

- `formulation`: The drag formulation, either `QuadraticFormulation()` (default) or `LinearFormulation()`.

# Keyword Arguments

- `coefficient`: The drag coefficient (required).
- `background_velocities`: Background velocities as a tuple `(U∞, V∞, W∞)` (default: `(0, 0, 0)`).
  These are added to the prognostic velocities when computing both the speed and the drag.


With `QuadraticFormulation()` (default), the drag is:
```math
τᵘ = - Cᴰ |U + U∞| (u + U∞)
```

With `LinearFormulation()` (Rayleigh friction), the drag is:
```math
τᵘ = - Cᴰ (u + U∞)
```

where `Cᴰ` is the drag coefficient, `|U + U∞| = √((u + U∞)² + (v + V∞)² + (w + W∞)²)` is the
total 3D speed including background velocities, and `(U∞, V∞, W∞)` are the background velocities.
The boundary-normal velocity component is zero due to the no-penetration condition.

This boundary condition can be applied to any of the six domain boundaries (west, east,
south, north, bottom, top) as well as immersed boundaries. The boundary side is automatically
determined during regularization.

See [`BulkDragFunction`](@ref) for details.

# Positional Arguments

- `formulation`: The drag formulation, either `QuadraticFormulation()` (default) or `LinearFormulation()`.

# Keyword Arguments

- `coefficient`: The drag coefficient (required).
- `direction`: The direction of the velocity component (`XDirection()`, `YDirection()`, or
               `ZDirection()`). If `nothing`, the direction is automatically inferred from
               the field location during boundary condition regularization.
- `background_velocities`: Background velocities as a tuple `(U∞, V∞, W∞)` (default: `(0, 0, 0)`).

# Examples

Create bulk drag boundary conditions for `u` and `v` at the domain bottom.
The direction is automatically inferred from the field location:

```jldoctest
using Oceananigans

drag = BulkDrag(coefficient=1e-3)
u_bcs = FieldBoundaryConditions(bottom=drag)
v_bcs = FieldBoundaryConditions(bottom=drag)

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid; boundary_conditions=(u=u_bcs, v=v_bcs))
model.velocities.u.boundary_conditions

# output
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: PeriodicBoundaryCondition
├── east: PeriodicBoundaryCondition
├── south: PeriodicBoundaryCondition
├── north: PeriodicBoundaryCondition
├── bottom: FluxBoundaryCondition: BulkDragFunction(QuadraticFormulation(), XDirection(), Cᴰ=0.001)
├── top: FluxBoundaryCondition: Nothing
└── immersed: Nothing
```

With immersed boundary conditions, apply drag only to the bottom facet
by using `ImmersedBoundaryCondition`. Here we also show how to implement linear drag:

```jldoctest
using Oceananigans

underlying_grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> -0.5))

# Apply to domain bottom and only the bottom facet of immersed boundaries
drag = BulkDrag(LinearFormulation(), coefficient=1e-3)
u_bcs = FieldBoundaryConditions(bottom=drag, immersed=ImmersedBoundaryCondition(bottom=drag))
v_bcs = FieldBoundaryConditions(bottom=drag, immersed=ImmersedBoundaryCondition(bottom=drag))

model = HydrostaticFreeSurfaceModel(; grid, boundary_conditions=(u=u_bcs, v=v_bcs))

# Verify the immersed BC has drag only on the bottom facet
model.velocities.u.boundary_conditions.immersed

# output
ImmersedBoundaryCondition:
├── west: Nothing
├── east: Nothing
├── south: Nothing
├── north: Nothing
├── bottom: FluxBoundaryCondition: BulkDragFunction(LinearFormulation(), XDirection(), Cᴰ=0.001)
└── top: Nothing
```

We can also apply three-dimensional drag to all facets:

```jldoctest three_d_drag
using Oceananigans

x = y = (-10, 10)
z = (0, 4)
mountain(x, y) = exp(-(x^2 + y^2) / 2)

underlying_grid = RectilinearGrid(size=(4, 4, 4); x, y, z, topology=(Bounded, Bounded, Bounded))
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(mountain))

# Apply to domain bottom and only the bottom facet of immersed boundaries
drag = BulkDrag(coefficient=1e-3)
u_bcs = FieldBoundaryConditions(south=drag, north=drag, bottom=drag, immersed=drag)
v_bcs = FieldBoundaryConditions(west=drag, east=drag, bottom=drag, immersed=drag)
w_bcs = FieldBoundaryConditions(south=drag, north=drag, west=drag, east=drag, immersed=drag)

model = HydrostaticFreeSurfaceModel(; grid, boundary_conditions=(u=u_bcs, v=v_bcs, w=w_bcs))

model.velocities.u.boundary_conditions

# output
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: OpenBoundaryCondition{Nothing}: Nothing
├── east: OpenBoundaryCondition{Nothing}: Nothing
├── south: FluxBoundaryCondition: BulkDragFunction(QuadraticFormulation(), XDirection(), Cᴰ=0.001)
├── north: FluxBoundaryCondition: BulkDragFunction(QuadraticFormulation(), XDirection(), Cᴰ=0.001)
├── bottom: FluxBoundaryCondition: BulkDragFunction(QuadraticFormulation(), XDirection(), Cᴰ=0.001)
├── top: FluxBoundaryCondition: Nothing
└── immersed: ImmersedBoundaryCondition with west=Nothing, east=Nothing, south=Flux, north=Flux, bottom=Flux, top=Flux
```

Notice that the syntax `immersed=drag` will add the drag condition to all non-normal facets for each velocity component,

```jldoctest three_d_drag
model.velocities.u.boundary_conditions.immersed

# output
ImmersedBoundaryCondition:
├── west: Nothing
├── east: Nothing
├── south: FluxBoundaryCondition: BulkDragFunction(QuadraticFormulation(), XDirection(), Cᴰ=0.001)
├── north: FluxBoundaryCondition: BulkDragFunction(QuadraticFormulation(), XDirection(), Cᴰ=0.001)
├── bottom: FluxBoundaryCondition: BulkDragFunction(QuadraticFormulation(), XDirection(), Cᴰ=0.001)
└── top: FluxBoundaryCondition: BulkDragFunction(QuadraticFormulation(), XDirection(), Cᴰ=0.001)
```
"""
BulkDrag(formulation=QuadraticFormulation(); kwargs...) =
    BoundaryCondition(Flux(), BulkDragFunction(formulation; kwargs...))

end # module
