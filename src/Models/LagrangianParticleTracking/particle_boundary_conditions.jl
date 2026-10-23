#####
##### Boundary conditions for Lagrangian particles
#####

"""
    ReflectingBoundary(restitution = 1.0)

A boundary condition that reflects Lagrangian particles back into the domain. A particle that
crosses the boundary by a distance `d` is placed a distance `restitution * d` inside the boundary.
"""
struct ReflectingBoundary{R}
    restitution :: R
end

ReflectingBoundary() = ReflectingBoundary(1.0)

"""
    AbsorbingBoundary()

A boundary condition that removes Lagrangian particles that cross the boundary.
"""
struct AbsorbingBoundary end

Base.summary(bc::ReflectingBoundary) = string("ReflectingBoundary(", bc.restitution, ")")
Base.summary(::AbsorbingBoundary) = "AbsorbingBoundary()"
Base.show(io::IO, bc::Union{ReflectingBoundary, AbsorbingBoundary}) = print(io, summary(bc))

@inline absorbs(::ReflectingBoundary) = false
@inline absorbs(::AbsorbingBoundary) = true

struct ParticleBoundaryConditions{W, E, S, N, B, T, I}
        west :: W
        east :: E
       south :: S
       north :: N
      bottom :: B
         top :: T
    immersed :: I
end

"""
    ParticleBoundaryConditions(default = ReflectingBoundary();
                               west = default, east = default,
                               south = default, north = default,
                               bottom = default, top = default,
                               immersed = default)

Return boundary conditions for `LagrangianParticles` on the `west`, `east`, `south`, `north`, `bottom`,
and `top` boundaries of a grid, and on `immersed` boundaries. Each boundary condition is either a
[`ReflectingBoundary`](@ref) or an [`AbsorbingBoundary`](@ref).

Boundary conditions apply only to boundaries in `Bounded` directions, and immersed boundary conditions
apply only on an `ImmersedBoundaryGrid`.

Examples
========

```jldoctest
julia> using Oceananigans

julia> ParticleBoundaryConditions(bottom = AbsorbingBoundary())
ParticleBoundaryConditions:
├── west: ReflectingBoundary(1.0)
├── east: ReflectingBoundary(1.0)
├── south: ReflectingBoundary(1.0)
├── north: ReflectingBoundary(1.0)
├── bottom: AbsorbingBoundary()
├── top: ReflectingBoundary(1.0)
└── immersed: ReflectingBoundary(1.0)
```
"""
function ParticleBoundaryConditions(default = ReflectingBoundary();
                                    west = default, east = default,
                                    south = default, north = default,
                                    bottom = default, top = default,
                                    immersed = default)

    return ParticleBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

const particle_boundary_sides = (:west, :east, :south, :north, :bottom, :top, :immersed)

boundary_conditions_tuple(bcs::ParticleBoundaryConditions) = Tuple(getproperty(bcs, side) for side in particle_boundary_sides)

absorbs(bcs::ParticleBoundaryConditions) = any(absorbs, boundary_conditions_tuple(bcs))

# Summarize as the equivalent constructor call, using the most common boundary condition as the default
function Base.summary(bcs::ParticleBoundaryConditions)
    conditions = boundary_conditions_tuple(bcs)
    default = argmax(bc -> count(==(bc), conditions), conditions)
    exceptions = [string(side, "=", summary(bc)) for (side, bc) in zip(particle_boundary_sides, conditions) if bc != default]
    keywords = isempty(exceptions) ? "" : string("; ", join(exceptions, ", "))
    return string("ParticleBoundaryConditions(", summary(default), keywords, ")")
end

function Base.show(io::IO, bcs::ParticleBoundaryConditions)
    print(io, "ParticleBoundaryConditions:", "\n",
              "├── west: ",     summary(bcs.west), "\n",
              "├── east: ",     summary(bcs.east), "\n",
              "├── south: ",    summary(bcs.south), "\n",
              "├── north: ",    summary(bcs.north), "\n",
              "├── bottom: ",   summary(bcs.bottom), "\n",
              "├── top: ",      summary(bcs.top), "\n",
              "└── immersed: ", summary(bcs.immersed))
end

#####
##### Enforcing boundary conditions
#####

# Functions for bouncing particles off walls to the right and left
@inline function bounce_left(x, xᴸ, xᴿ, Cʳ)
    xᵢ = xᴿ - Cʳ * (x - xᴿ)
    # Keep the particle from bouncing so far left it leaves the domain
    return ifelse(xᵢ < xᴸ, xᴸ, xᵢ)
end

@inline function bounce_right(x, xᴸ, xᴿ, Cʳ)
    xᵢ = xᴸ + Cʳ * (xᴸ - x)
    # Keep the particle from bouncing so far right it leaves the domain
    return ifelse(xᵢ > xᴿ, xᴿ, xᵢ)
end

# Position of a particle at `x` that crossed the right boundary `xᴿ`
@inline right_boundary_position(bc::ReflectingBoundary, x, xᴸ, xᴿ) = bounce_left(x, xᴸ, xᴿ, bc.restitution)
@inline right_boundary_position(::AbsorbingBoundary, x, xᴸ, xᴿ) = xᴿ

# Position of a particle at `x` that crossed the left boundary `xᴸ`
@inline left_boundary_position(bc::ReflectingBoundary, x, xᴸ, xᴿ) = bounce_right(x, xᴸ, xᴿ, bc.restitution)
@inline left_boundary_position(::AbsorbingBoundary, x, xᴸ, xᴿ) = xᴸ

"""
$(TYPEDSIGNATURES)

Return the new position of a particle at `x` in the Bounded interval `(xᴸ, xᴿ)`
with boundary conditions `left_bc` and `right_bc`, and whether the particle was absorbed.
Absorbed particles are placed on the boundary they crossed.
"""
@inline function enforce_boundary_conditions(::Bounded, x, xᴸ, xᴿ, left_bc, right_bc)
    crossed_right = x > xᴿ
    crossed_left = x < xᴸ

    x⁺ = ifelse(crossed_right, right_boundary_position(right_bc, x, xᴸ, xᴿ),
         ifelse(crossed_left,  left_boundary_position(left_bc, x, xᴸ, xᴿ), x))

    absorbed = (crossed_right & absorbs(right_bc)) | (crossed_left & absorbs(left_bc))

    return x⁺, absorbed
end

"""
$(TYPEDSIGNATURES)

Return the new position of a particle at `x` in the Periodic interval `(xᴸ, xᴿ)`.
Particles are never absorbed at periodic boundaries.
"""
@inline function enforce_boundary_conditions(::Periodic, x, xᴸ, xᴿ, left_bc, right_bc)
    x⁺ = ifelse(x > xᴿ, xᴸ + mod(x - xᴿ, xᴿ - xᴸ),
         ifelse(x < xᴸ, xᴿ - mod(xᴸ - x, xᴿ - xᴸ), x))
    return x⁺, false
end

"""
$(TYPEDSIGNATURES)

Do nothing on Flat dimensions.
"""
@inline enforce_boundary_conditions(::Flat, x, xᴸ, xᴿ, left_bc, right_bc) = x, false
