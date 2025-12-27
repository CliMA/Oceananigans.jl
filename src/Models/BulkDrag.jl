"""
    BulkDrag

Module for implementing quadratic drag boundary conditions based on similarity theory.
Provides a convenient interface for specifying drag forces on velocity fields at
solid boundaries (bottom, immersed boundaries, etc.)

# Background

Similarity theory supposes that shear is a function of only distance `d` from a wall:

    ∂_d u = u★ / (ϰ d)

where u★ is the friction velocity (τ = -u★²), ϰ is the von Karman constant.
Integrating from the roughness length ℓ gives:

    u(d) = (u★ / ϰ) log(d / ℓ)

Inverting for the stress τ at a particular distance d₀:

    τ = -u★² = -cᴰ |u(d₀)| u(d₀)

where the drag coefficient is cᴰ = (ϰ / log(d₀ / ℓ))².

# Usage

```julia
bulk_drag = BulkDrag(roughness_length=1e-4)
u_bcs, v_bcs, w_bcs = drag_boundary_conditions(grid, bulk_drag)

# For hydrostatic models (no w drag):
u_bcs, v_bcs = drag_boundary_conditions(grid, bulk_drag; include_vertical_velocity=false)

# Use as immersed boundary condition:
u_immersed_bc = ImmersedBoundaryCondition(bulk_drag, grid, :u)
```
"""
module BulkDragModule

export BulkDrag, drag_boundary_conditions, drag_immersed_boundary_conditions

using Oceananigans.Grids: AbstractGrid, minimum_xspacing, minimum_yspacing, minimum_zspacing
using Oceananigans.Grids: XFlatGrid, YFlatGrid, ZFlatGrid, topology
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxzᶠᵃᶜ, ℑxzᶜᵃᶠ, ℑyzᵃᶠᶜ, ℑyzᵃᶜᶠ
using Oceananigans.BoundaryConditions: FluxBoundaryCondition
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryCondition

#####
##### BulkDrag struct
#####

"""
    struct BulkDrag{FT, DC}

Parameters for computing quadratic drag based on similarity theory.

# Fields
- `roughness_length`: Inner layer thickness ℓ (often called roughness length) [m]
- `von_karman_constant`: Von Karman constant ϰ ≈ 0.4
- `drag_coefficient`: Optionally specify drag coefficient directly (overrides roughness-based calculation)

# Constructors

    BulkDrag(; roughness_length = 1e-4,
               von_karman_constant = 0.4,
               drag_coefficient = nothing)

If `drag_coefficient` is specified, it is used directly. Otherwise, the drag coefficient
is computed from `roughness_length` and `von_karman_constant` based on the grid spacing.
"""
struct BulkDrag{FT, DC}
    roughness_length :: FT
    von_karman_constant :: FT
    drag_coefficient :: DC
end

"""
    BulkDrag(FT::DataType = Float64;
             roughness_length = 1e-4,
             von_karman_constant = 0.4,
             drag_coefficient = nothing)

Construct a `BulkDrag` parameterization for computing quadratic drag forces at boundaries.

# Arguments
- `FT`: Floating point type (default: Float64)

# Keyword Arguments
- `roughness_length`: The roughness length ℓ in meters (default: 1e-4)
- `von_karman_constant`: Von Karman constant ϰ (default: 0.4)
- `drag_coefficient`: Optionally specify drag coefficient directly. If `nothing`,
                      coefficient is computed from similarity theory using grid spacing.

# Examples

```jldoctest
julia> using Oceananigans.Models: BulkDrag

julia> bulk_drag = BulkDrag(roughness_length=1e-4)
BulkDrag{Float64}:
├── roughness_length: 0.0001
├── von_karman_constant: 0.4
└── drag_coefficient: nothing (computed from similarity theory)

julia> bulk_drag_fixed = BulkDrag(drag_coefficient=0.003)
BulkDrag{Float64}:
├── roughness_length: 0.0001
├── von_karman_constant: 0.4
└── drag_coefficient: 0.003
```
"""
function BulkDrag(FT::DataType = Float64;
                  roughness_length = 1e-4,
                  von_karman_constant = 0.4,
                  drag_coefficient = nothing)

    return BulkDrag(convert(FT, roughness_length),
                    convert(FT, von_karman_constant),
                    isnothing(drag_coefficient) ? nothing : convert(FT, drag_coefficient))
end

function Base.summary(drag::BulkDrag)
    if isnothing(drag.drag_coefficient)
        return string("BulkDrag(ℓ=", drag.roughness_length, ", ϰ=", drag.von_karman_constant, ")")
    else
        return string("BulkDrag(cᴰ=", drag.drag_coefficient, ")")
    end
end

function Base.show(io::IO, drag::BulkDrag{FT}) where FT
    cᴰ_str = isnothing(drag.drag_coefficient) ? "nothing (computed from similarity theory)" : string(drag.drag_coefficient)
    print(io, "BulkDrag{$FT}:", '\n',
              "├── roughness_length: ", drag.roughness_length, '\n',
              "├── von_karman_constant: ", drag.von_karman_constant, '\n',
              "└── drag_coefficient: ", cᴰ_str)
end

#####
##### Compute drag coefficient from similarity theory
#####

"""
    compute_drag_coefficient(drag::BulkDrag, grid, direction)

Compute the drag coefficient from similarity theory:

    cᴰ = (ϰ / log(d₀ / ℓ))²

where d₀ is half the grid spacing in the direction normal to the boundary,
ϰ is the von Karman constant, and ℓ is the roughness length.
"""
function compute_drag_coefficient(drag::BulkDrag, grid, direction::Symbol)
    !isnothing(drag.drag_coefficient) && return drag.drag_coefficient

    ϰ = drag.von_karman_constant
    ℓ = drag.roughness_length

    # Distance to wall is half the grid spacing
    if direction === :x
        d₀ = minimum_xspacing(grid) / 2
    elseif direction === :y
        d₀ = minimum_yspacing(grid) / 2
    elseif direction === :z
        d₀ = minimum_zspacing(grid) / 2
    else
        error("Invalid direction: $direction. Must be :x, :y, or :z.")
    end

    d₀ ≤ ℓ && @warn "Grid spacing ($d₀) is less than or equal to roughness length ($ℓ)!"
    
    cᴰ = (ϰ / log(d₀ / ℓ))^2
    return cᴰ
end

#####
##### Helper functions for computing speed at staggered locations
#####

# Square of a field value
@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

# Speed at u-points (Face, Center, Center)
@inline function speedᶠᶜᶜ(i, j, k, grid, u, v, w)
    u² = @inbounds u[i, j, k]^2
    v² = ℑxyᶠᶜᵃ(i, j, k, grid, ϕ², v)
    w² = ℑxzᶠᵃᶜ(i, j, k, grid, ϕ², w)
    return sqrt(u² + v² + w²)
end

# Speed at v-points (Center, Face, Center)
@inline function speedᶜᶠᶜ(i, j, k, grid, u, v, w)
    u² = ℑxyᶜᶠᵃ(i, j, k, grid, ϕ², u)
    v² = @inbounds v[i, j, k]^2
    w² = ℑyzᵃᶠᶜ(i, j, k, grid, ϕ², w)
    return sqrt(u² + v² + w²)
end

# Speed at w-points (Center, Center, Face)
@inline function speedᶜᶜᶠ(i, j, k, grid, u, v, w)
    u² = ℑxzᶜᵃᶠ(i, j, k, grid, ϕ², u)
    v² = ℑyzᵃᶜᶠ(i, j, k, grid, ϕ², v)
    w² = @inbounds w[i, j, k]^2
    return sqrt(u² + v² + w²)
end

# 2D speed (for hydrostatic models without w)
@inline function speed_xyᶠᶜᶜ(i, j, k, grid, u, v)
    u² = @inbounds u[i, j, k]^2
    v² = ℑxyᶠᶜᵃ(i, j, k, grid, ϕ², v)
    return sqrt(u² + v²)
end

@inline function speed_xyᶜᶠᶜ(i, j, k, grid, u, v)
    u² = ℑxyᶜᶠᵃ(i, j, k, grid, ϕ², u)
    v² = @inbounds v[i, j, k]^2
    return sqrt(u² + v²)
end

#####
##### Drag flux functions for boundary conditions
#####

# Parameters struct for drag flux functions
struct BulkDragParameters{FT}
    drag_coefficient :: FT
end

# 3D drag flux functions (include w in speed calculation)
@inline function u_bulk_drag_flux(i, j, grid, clock, fields, p::BulkDragParameters)
    k = 1 # bottom boundary
    cᴰ = p.drag_coefficient
    speed = speedᶠᶜᶜ(i, j, k, grid, fields.u, fields.v, fields.w)
    return @inbounds -cᴰ * fields.u[i, j, k] * speed
end

@inline function v_bulk_drag_flux(i, j, grid, clock, fields, p::BulkDragParameters)
    k = 1 # bottom boundary
    cᴰ = p.drag_coefficient
    speed = speedᶜᶠᶜ(i, j, k, grid, fields.u, fields.v, fields.w)
    return @inbounds -cᴰ * fields.v[i, j, k] * speed
end

@inline function w_bulk_drag_flux(i, j, grid, clock, fields, p::BulkDragParameters)
    k = 1 # bottom boundary
    cᴰ = p.drag_coefficient
    speed = speedᶜᶜᶠ(i, j, k, grid, fields.u, fields.v, fields.w)
    return @inbounds -cᴰ * fields.w[i, j, k] * speed
end

# 2D drag flux functions (exclude w from speed calculation, for hydrostatic models)
@inline function u_bulk_drag_flux_xy(i, j, grid, clock, fields, p::BulkDragParameters)
    k = 1 # bottom boundary
    cᴰ = p.drag_coefficient
    speed = speed_xyᶠᶜᶜ(i, j, k, grid, fields.u, fields.v)
    return @inbounds -cᴰ * fields.u[i, j, k] * speed
end

@inline function v_bulk_drag_flux_xy(i, j, grid, clock, fields, p::BulkDragParameters)
    k = 1 # bottom boundary
    cᴰ = p.drag_coefficient
    speed = speed_xyᶜᶠᶜ(i, j, k, grid, fields.u, fields.v)
    return @inbounds -cᴰ * fields.v[i, j, k] * speed
end

#####
##### 3D drag flux functions for immersed boundaries
##### These take (i, j, k) as arguments instead of just (i, j)
#####

@inline function u_bulk_drag_flux_3d(i, j, k, grid, clock, fields, p::BulkDragParameters)
    cᴰ = p.drag_coefficient
    speed = speedᶠᶜᶜ(i, j, k, grid, fields.u, fields.v, fields.w)
    return @inbounds -cᴰ * fields.u[i, j, k] * speed
end

@inline function v_bulk_drag_flux_3d(i, j, k, grid, clock, fields, p::BulkDragParameters)
    cᴰ = p.drag_coefficient
    speed = speedᶜᶠᶜ(i, j, k, grid, fields.u, fields.v, fields.w)
    return @inbounds -cᴰ * fields.v[i, j, k] * speed
end

@inline function w_bulk_drag_flux_3d(i, j, k, grid, clock, fields, p::BulkDragParameters)
    cᴰ = p.drag_coefficient
    speed = speedᶜᶜᶠ(i, j, k, grid, fields.u, fields.v, fields.w)
    return @inbounds -cᴰ * fields.w[i, j, k] * speed
end

# 2D versions for hydrostatic models
@inline function u_bulk_drag_flux_xy_3d(i, j, k, grid, clock, fields, p::BulkDragParameters)
    cᴰ = p.drag_coefficient
    speed = speed_xyᶠᶜᶜ(i, j, k, grid, fields.u, fields.v)
    return @inbounds -cᴰ * fields.u[i, j, k] * speed
end

@inline function v_bulk_drag_flux_xy_3d(i, j, k, grid, clock, fields, p::BulkDragParameters)
    cᴰ = p.drag_coefficient
    speed = speed_xyᶜᶠᶜ(i, j, k, grid, fields.u, fields.v)
    return @inbounds -cᴰ * fields.v[i, j, k] * speed
end

#####
##### Convenience function to create boundary conditions
#####

"""
    drag_boundary_conditions(grid, bulk_drag::BulkDrag;
                             boundary = :bottom,
                             include_vertical_velocity = true)

Create flux boundary conditions for velocity components that apply quadratic drag.

# Arguments
- `grid`: The model grid
- `bulk_drag`: A `BulkDrag` instance specifying the drag parameterization

# Keyword Arguments
- `boundary`: Which boundary to apply drag to. Options: `:bottom`, `:top`, `:west`, `:east`, 
              `:south`, `:north`. Default: `:bottom`.
- `include_vertical_velocity`: Whether to include vertical velocity in speed calculation and
                               return w boundary condition. Set to `false` for hydrostatic models.
                               Default: `true`.

# Returns
- If `include_vertical_velocity=true`: `(u_bc, v_bc, w_bc)` tuple of FluxBoundaryConditions
- If `include_vertical_velocity=false`: `(u_bc, v_bc)` tuple of FluxBoundaryConditions

# Examples

```julia
using Oceananigans
using Oceananigans.Models: BulkDrag, drag_boundary_conditions

grid = RectilinearGrid(size=(10, 10, 10), extent=(1, 1, 1))
bulk_drag = BulkDrag(roughness_length=1e-4)

# For nonhydrostatic model (includes w):
u_bc, v_bc, w_bc = drag_boundary_conditions(grid, bulk_drag)

# For hydrostatic model (excludes w):
u_bc, v_bc = drag_boundary_conditions(grid, bulk_drag; include_vertical_velocity=false)
```
"""
function drag_boundary_conditions(grid, bulk_drag::BulkDrag;
                                  boundary::Symbol = :bottom,
                                  include_vertical_velocity::Bool = true)

    # Determine the direction normal to the boundary
    if boundary in (:bottom, :top)
        direction = :z
    elseif boundary in (:west, :east)
        direction = :x
    elseif boundary in (:south, :north)
        direction = :y
    else
        error("Invalid boundary: $boundary. Must be :bottom, :top, :west, :east, :south, or :north.")
    end

    cᴰ = compute_drag_coefficient(bulk_drag, grid, direction)
    parameters = BulkDragParameters(cᴰ)

    if include_vertical_velocity
        u_bc = FluxBoundaryCondition(u_bulk_drag_flux, discrete_form=true, parameters=parameters)
        v_bc = FluxBoundaryCondition(v_bulk_drag_flux, discrete_form=true, parameters=parameters)
        w_bc = FluxBoundaryCondition(w_bulk_drag_flux, discrete_form=true, parameters=parameters)
        return (u_bc, v_bc, w_bc)
    else
        u_bc = FluxBoundaryCondition(u_bulk_drag_flux_xy, discrete_form=true, parameters=parameters)
        v_bc = FluxBoundaryCondition(v_bulk_drag_flux_xy, discrete_form=true, parameters=parameters)
        return (u_bc, v_bc)
    end
end

"""
    drag_immersed_boundary_conditions(grid, bulk_drag::BulkDrag;
                                      include_vertical_velocity = true)

Create immersed boundary conditions for velocity components that apply quadratic drag
on all immersed boundary interfaces.

# Arguments
- `grid`: The model grid (typically an ImmersedBoundaryGrid)
- `bulk_drag`: A `BulkDrag` instance specifying the drag parameterization

# Keyword Arguments
- `include_vertical_velocity`: Whether to include vertical velocity in speed calculation and
                               return w boundary condition. Set to `false` for hydrostatic models.
                               Default: `true`.

# Returns
- If `include_vertical_velocity=true`: `(u_ibc, v_ibc, w_ibc)` tuple of ImmersedBoundaryConditions
- If `include_vertical_velocity=false`: `(u_ibc, v_ibc)` tuple of ImmersedBoundaryConditions

# Examples

```julia
using Oceananigans
using Oceananigans.Models: BulkDrag, drag_immersed_boundary_conditions
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom

grid = RectilinearGrid(size=(10, 10, 10), extent=(1, 1, 1))
bottom(x, y) = -0.5 + 0.1 * sin(2π * x)
ibg = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

bulk_drag = BulkDrag(roughness_length=1e-4)
u_ibc, v_ibc, w_ibc = drag_immersed_boundary_conditions(ibg, bulk_drag)
```
"""
function drag_immersed_boundary_conditions(grid, bulk_drag::BulkDrag;
                                           include_vertical_velocity::Bool = true)

    # Use minimum spacing for drag coefficient (conservative choice)
    min_spacing = min(minimum_xspacing(grid), minimum_yspacing(grid), minimum_zspacing(grid))
    
    ϰ = bulk_drag.von_karman_constant
    ℓ = bulk_drag.roughness_length
    
    if !isnothing(bulk_drag.drag_coefficient)
        cᴰ = bulk_drag.drag_coefficient
    else
        d₀ = min_spacing / 2
        d₀ ≤ ℓ && @warn "Grid spacing ($d₀) is less than or equal to roughness length ($ℓ)!"
        cᴰ = (ϰ / log(d₀ / ℓ))^2
    end

    parameters = BulkDragParameters(cᴰ)

    if include_vertical_velocity
        u_flux = FluxBoundaryCondition(u_bulk_drag_flux_3d, discrete_form=true, parameters=parameters)
        v_flux = FluxBoundaryCondition(v_bulk_drag_flux_3d, discrete_form=true, parameters=parameters)
        w_flux = FluxBoundaryCondition(w_bulk_drag_flux_3d, discrete_form=true, parameters=parameters)

        u_ibc = ImmersedBoundaryCondition(west=u_flux, east=u_flux, south=u_flux, north=u_flux, bottom=u_flux, top=u_flux)
        v_ibc = ImmersedBoundaryCondition(west=v_flux, east=v_flux, south=v_flux, north=v_flux, bottom=v_flux, top=v_flux)
        w_ibc = ImmersedBoundaryCondition(west=w_flux, east=w_flux, south=w_flux, north=w_flux, bottom=w_flux, top=w_flux)

        return (u_ibc, v_ibc, w_ibc)
    else
        u_flux = FluxBoundaryCondition(u_bulk_drag_flux_xy_3d, discrete_form=true, parameters=parameters)
        v_flux = FluxBoundaryCondition(v_bulk_drag_flux_xy_3d, discrete_form=true, parameters=parameters)

        u_ibc = ImmersedBoundaryCondition(west=u_flux, east=u_flux, south=u_flux, north=u_flux, bottom=u_flux, top=u_flux)
        v_ibc = ImmersedBoundaryCondition(west=v_flux, east=v_flux, south=v_flux, north=v_flux, bottom=v_flux, top=v_flux)

        return (u_ibc, v_ibc)
    end
end

end # module

