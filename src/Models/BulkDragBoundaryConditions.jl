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

using Adapt: Adapt
using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans.Architectures: Architectures, on_architecture
using Oceananigans.BoundaryConditions: BoundaryConditions, BoundaryCondition, Flux,
                                       LeftBoundary, RightBoundary, Bottom, Top, ImmersedFacet,
                                       ExplicitTimeDiscretization, IMEXFluxTimeDiscretization
using Oceananigans.Grids: AbstractGrid, XDirection, YDirection, ZDirection, Face
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxzᶠᵃᶜ, ℑyzᵃᶠᶜ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶠ

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
##### The drag coefficient λ, such that the drag flux on the tangential velocity is λ (u + U∞)
#####

# Quadratic drag: λ = -Cᴰ |U|
@inline x_drag_coefficient(i, j, k, grid, ::QuadraticFormulation, fields, Cᴰ, U∞, V∞, W∞) = - Cᴰ * sqrt(speed²ᶠᶜᶜ(i, j, k, grid, fields, U∞, V∞, W∞))
@inline y_drag_coefficient(i, j, k, grid, ::QuadraticFormulation, fields, Cᴰ, U∞, V∞, W∞) = - Cᴰ * sqrt(speed²ᶜᶠᶜ(i, j, k, grid, fields, U∞, V∞, W∞))
@inline z_drag_coefficient(i, j, k, grid, ::QuadraticFormulation, fields, Cᴰ, U∞, V∞, W∞) = - Cᴰ * sqrt(speed²ᶜᶜᶠ(i, j, k, grid, fields, U∞, V∞, W∞))

# Linear drag (Rayleigh friction): λ = -Cᴰ
@inline x_drag_coefficient(i, j, k, grid, ::LinearFormulation, fields, Cᴰ, U∞, V∞, W∞) = - Cᴰ
@inline y_drag_coefficient(i, j, k, grid, ::LinearFormulation, fields, Cᴰ, U∞, V∞, W∞) = - Cᴰ
@inline z_drag_coefficient(i, j, k, grid, ::LinearFormulation, fields, Cᴰ, U∞, V∞, W∞) = - Cᴰ

const XDBDF = XDirectionBulkDragFunction
const YDBDF = YDirectionBulkDragFunction
const ZDBDF = ZDirectionBulkDragFunction

@inline drag_coefficient(i, j, k, grid, df::XDBDF, fields) = x_drag_coefficient(i, j, k, grid, df.formulation, fields, df.coefficient, df.background_velocities...)
@inline drag_coefficient(i, j, k, grid, df::YDBDF, fields) = y_drag_coefficient(i, j, k, grid, df.formulation, fields, df.coefficient, df.background_velocities...)
@inline drag_coefficient(i, j, k, grid, df::ZDBDF, fields) = z_drag_coefficient(i, j, k, grid, df.formulation, fields, df.coefficient, df.background_velocities...)

@inline tangential_velocity(i, j, k, grid, ::XDBDF, fields) = @inbounds fields.u[i, j, k]
@inline tangential_velocity(i, j, k, grid, ::YDBDF, fields) = @inbounds fields.v[i, j, k]
@inline tangential_velocity(i, j, k, grid, ::ZDBDF, fields) = @inbounds fields.w[i, j, k]

@inline background_velocity(df::XDBDF) = df.background_velocities[1]
@inline background_velocity(df::YDBDF) = df.background_velocities[2]
@inline background_velocity(df::ZDBDF) = df.background_velocities[3]

# The drag flux λ (u + U∞) along the inward-pointing boundary normal
@inline function bulk_drag(i, j, k, grid, df::BulkDragFunction, fields)
    λ  = drag_coefficient(i, j, k, grid, df, fields)
    u  = tangential_velocity(i, j, k, grid, df, fields)
    U∞ = background_velocity(df)
    return λ * (u + U∞)
end

#####
##### Domain boundaries: map the two boundary indices to (i, j, k) and orient the flux
#####

@inline boundary_index(::LeftBoundary,  N) = 1
@inline boundary_index(::RightBoundary, N) = N

# `dim` is the boundary-normal direction: the boundary index replaces the missing one
@inline boundary_ijk(::Val{1}, side, j, k, grid) = (boundary_index(side, grid.Nx), j, k)
@inline boundary_ijk(::Val{2}, side, i, k, grid) = (i, boundary_index(side, grid.Ny), k)
@inline boundary_ijk(::Val{3}, side, i, j, grid) = (i, j, boundary_index(side, grid.Nz))

# A flux on a left boundary (west, south, bottom) enters the tendency as +J/Δ, and on a right
# boundary (east, north, top) as -J/Δ. The drag `λ (u + U∞)` is a sink along the inward normal,
# so on a right boundary it must be flipped to oppose the flow rather than accelerate it.
@inline boundary_sign(::LeftBoundary)  = 1
@inline boundary_sign(::RightBoundary) = -1

@inline function BoundaryConditions.getbc(df::BulkDragFunction, a::Integer, b::Integer,
                                          grid::AbstractGrid, clock, fields, args...)
    i, j, k = boundary_ijk(df.dim, df.side, a, b, grid)
    return boundary_sign(df.side) * bulk_drag(i, j, k, grid, df, fields)
end

#####
##### Immersed boundaries: (i, j, k) is the boundary-adjacent cell and every facet flux points inward
#####

@inline BoundaryConditions.getbc(df::BulkDragFunction, i::Integer, j::Integer, k::Integer,
                                 grid::AbstractGrid, clock, fields, args...) = bulk_drag(i, j, k, grid, df, fields)

#####
##### Implicit-explicit time discretization: the drag λ (u + U∞) is affine in the tangential velocity, so
##### the explicit part is `λ U∞` and the linear part `λ u` goes into the vertical tridiagonal solver.
##### For the quadratic formulation the speed |U| in λ is lagged (evaluated when the diagonal is built).
#####

@inline boundary_sign(::Top) = -1
@inline boundary_sign(::Union{Bottom, ImmersedFacet}) = 1

@inline BoundaryConditions.implicit_flux_coefficient(df::BulkDragFunction, boundary, i, j, k, grid, ϕ, clock, fields, args...) =
    boundary_sign(boundary) * drag_coefficient(i, j, k, grid, df, fields)

@inline BoundaryConditions.explicit_flux(df::BulkDragFunction, boundary, i, j, k, grid, ϕ, clock, fields, args...) =
    BoundaryConditions.implicit_flux_coefficient(df, boundary, i, j, k, grid, ϕ, clock, fields) * background_velocity(df)

#####
##### Type alias for FluxBoundaryCondition with BulkDragFunction
#####

const BulkDragBoundaryCondition = BoundaryCondition{<:Flux, <:BulkDragFunction}

#####
##### Regularization: infer direction from field location and set Side
#####

"""
$(TYPEDSIGNATURES)

Regularize a `BulkDragFunction` by:
1. Inferring the direction from the field location if not specified
2. Setting the boundary side and dimension from the regularization context
3. Converting numeric parameters to the grid's float type

The direction is inferred as follows:
- If `loc[1] == Face`, the field is a u-velocity → `XDirection()`
- If `loc[2] == Face`, the field is a v-velocity → `YDirection()`
- If `loc[3] == Face`, the field is a w-velocity → `ZDirection()`

The dimension `dim` indicates which axis the boundary is normal to:
- `dim=1`: x-normal boundary (west/east)
- `dim=2`: y-normal boundary (south/north)
- `dim=3`: z-normal boundary (bottom/top)
"""
function BoundaryConditions.regularize_boundary_condition(df::BulkDragFunction, grid, loc, dim, Side, field_names)
    direction = infer_direction(df.direction, loc)
    FT = eltype(grid)
    coefficient = regularize_parameter(FT, df.coefficient)
    background_velocities = map(U -> regularize_parameter(FT, U), df.background_velocities)

    # Side() instantiates the Side type (e.g., LeftBoundary → LeftBoundary())
    # Val{dim} is used for type dispatch in getbc methods
    return BulkDragFunction(direction, Side(), Val{dim}(), df.formulation, coefficient, background_velocities)
end

infer_direction(direction, loc) = direction

function infer_direction(::Nothing, loc)
    if loc[1] isa Face
        return XDirection()
    elseif loc[2] isa Face
        return YDirection()
    elseif loc[3] isa Face
        return ZDirection()
    else
        error("Cannot infer BulkDragFunction direction for field at location $loc. " *
              "Please specify direction explicitly.")
    end
end

regularize_parameter(FT, p::Number) = convert(FT, p)
regularize_parameter(FT, p) = p

#####
##### Convenient constructor
#####

"""
    BulkDrag(formulation=QuadraticFormulation(); coefficient, background_velocities=(0, 0, 0),
             time_discretization=ExplicitTimeDiscretization())

Create a `FluxBoundaryCondition` for velocity drag on any boundary.

With `QuadraticFormulation()` (default), the drag is:
```math
τᵘ = - C^D |U + U_∞| (u + U_∞)
```

With `LinearFormulation()` (Rayleigh friction), the drag is:
```math
τᵘ = - C^D (u + U_∞)
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
  These are added to the prognostic velocities when computing both the speed and the drag.
- `time_discretization`: Either `ExplicitTimeDiscretization()` (default), which integrates the drag
                         through the tendency, or `IMEXFluxTimeDiscretization()`, which integrates
                         the linear part of the drag implicitly (see below).

# Implicit-explicit time discretization

The drag is affine in the tangential velocity, ``τ = λ (u + U_∞)`` with ``λ = -C^D |U + U_∞|`` for the
quadratic formulation and ``λ = -C^D`` for the linear one. With `time_discretization = IMEXFluxTimeDiscretization()`,
the explicit part ``λ U_∞`` is integrated through the tendency while the linear part ``λ u`` is embedded in the
vertical tridiagonal solver, which removes the time step restriction ``C^D |U| Δt / Δz < 2`` of the explicit
treatment. For the quadratic formulation the speed ``|U + U_∞|`` in ``λ`` is evaluated from the current velocities
when the implicit solve is built, so the drag is linearized about the current speed.

Like every [`IMEXFluxTimeDiscretization`](@ref), this is supported only on the `bottom` and `top` boundaries and on
the `bottom` and `top` facets of an [`ImmersedBoundaryCondition`](@ref).

```jldoctest
using Oceananigans

drag = BulkDrag(coefficient=1e-3, time_discretization=IMEXFluxTimeDiscretization())
u_bcs = FieldBoundaryConditions(bottom=drag)

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
model = HydrostaticFreeSurfaceModel(grid; boundary_conditions=(; u=u_bcs))
model.velocities.u.boundary_conditions.bottom

# output
IMEXFluxBoundaryCondition: BulkDragFunction(QuadraticFormulation(), XDirection(), Cᴰ=0.001)
```

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

model = HydrostaticFreeSurfaceModel(grid; boundary_conditions=(u=u_bcs, v=v_bcs))

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

model = HydrostaticFreeSurfaceModel(grid; boundary_conditions=(u=u_bcs, v=v_bcs, w=w_bcs))

model.velocities.u.boundary_conditions

# output
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: NormalFlowBoundaryCondition{Nothing}: Nothing
├── east: NormalFlowBoundaryCondition{Nothing}: Nothing
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
function BulkDrag(formulation=QuadraticFormulation(); time_discretization=ExplicitTimeDiscretization(), kwargs...)
    validate_drag_time_discretization(time_discretization)
    return BoundaryCondition(Flux(time_discretization), BulkDragFunction(formulation; kwargs...))
end

validate_drag_time_discretization(td) = nothing

validate_drag_time_discretization(td::IMEXFluxTimeDiscretization) =
    isnothing(td.implicit_coefficient) ||
        throw(ArgumentError("BulkDrag computes its own implicit coefficient; pass IMEXFluxTimeDiscretization() without one"))

end # module
