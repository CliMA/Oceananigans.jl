module LagrangianParticleTracking

export LagrangianParticles

using Printf
using Adapt
using KernelAbstractions
using StructArrays

using Oceananigans.Grids
using Oceananigans.ImmersedBoundaries

using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.Grids: AbstractUnderlyingGrid, AbstractGrid, hack_cosd
using Oceananigans.Grids: XFlatGrid, YFlatGrid, ZFlatGrid
using Oceananigans.Grids: XYFlatGrid, YZFlatGrid, XZFlatGrid
using Oceananigans.ImmersedBoundaries: immersed_cell
using Oceananigans.Architectures: device, architecture
using Oceananigans.Fields: interpolate, datatuple, compute!, location
using Oceananigans.Fields: fractional_indices
using Oceananigans.TimeSteppers: AbstractLagrangianParticles
using Oceananigans.Utils: prettysummary, launch!, SumOfArrays

import Oceananigans.TimeSteppers: step_lagrangian_particles!

import Base: size, length, show

abstract type AbstractParticle end

struct Particle{T} <: AbstractParticle
    x :: T
    y :: T
    z :: T
end

Base.show(io::IO, p::Particle) = print(io, "Particle at (",
                                       @sprintf("%-8s", prettysummary(p.x, true) * ", "),
                                       @sprintf("%-8s", prettysummary(p.y, true) * ", "),
                                       @sprintf("%-8s", prettysummary(p.z, true) * ")"))

struct LagrangianParticles{P, R, T, D, Π} <: AbstractLagrangianParticles
        properties :: P
       restitution :: R
    tracked_fields :: T
          dynamics :: D
        parameters :: Π
end

@inline no_dynamics(args...) = nothing

"""
    LagrangianParticles(; x, y, z, restitution=1.0, dynamics=no_dynamics, parameters=nothing)

Construct some `LagrangianParticles` that can be passed to a model. The particles will have initial locations
`x`, `y`, and `z`. The coefficient of restitution for particle-wall collisions is specified by `restitution`.

`dynamics` is a function of `(lagrangian_particles, model, Δt)` that is called prior to advecting particles.
`parameters` can be accessed inside the `dynamics` function.
"""
function LagrangianParticles(; x, y, z, restitution=1.0, dynamics=no_dynamics, parameters=nothing)
    size(x) == size(y) == size(z) ||
        throw(ArgumentError("x, y, z must all have the same size!"))

    (ndims(x) == 1 && ndims(y) == 1 && ndims(z) == 1) ||
        throw(ArgumentError("x, y, z must have dimension 1 but ndims=($(ndims(x)), $(ndims(y)), $(ndims(z)))"))

    particles = StructArray{Particle}((x, y, z))

    return LagrangianParticles(particles; restitution, dynamics, parameters)
end

"""
    LagrangianParticles(particles::StructArray; restitution=1.0, tracked_fields::NamedTuple=NamedTuple(), dynamics=no_dynamics)

Construct some `LagrangianParticles` that can be passed to a model. The `particles` should be a `StructArray`
and can contain custom fields. The coefficient of restitution for particle-wall collisions is specified by `restitution`.

A number of `tracked_fields` may be passed in as a `NamedTuple` of fields. Each particle will track the value of each
field. Each tracked field must have a corresponding particle property. So if `T` is a tracked field, then `T` must also
be a custom particle property.

`dynamics` is a function of `(lagrangian_particles, model, Δt)` that is called prior to advecting particles.
`parameters` can be accessed inside the `dynamics` function.
"""
function LagrangianParticles(particles::StructArray;
                             restitution = 1.0,
                             tracked_fields::NamedTuple=NamedTuple(),
                             dynamics = no_dynamics,
                             parameters = nothing)

    for (field_name, tracked_field) in pairs(tracked_fields)
        field_name in propertynames(particles) ||
            throw(ArgumentError("$field_name is a tracked field but $(eltype(particles)) has no $field_name field! " *
                                "You might have to define your own particle type."))
    end

    return LagrangianParticles(particles, restitution, tracked_fields, dynamics, parameters)
end

size(lagrangian_particles::LagrangianParticles) = size(lagrangian_particles.properties)
length(lagrangian_particles::LagrangianParticles) = length(lagrangian_particles.properties)

Base.summary(particles::LagrangianParticles) =
    string(length(particles), " LagrangianParticles with eltype ", nameof(eltype(particles.properties)),
           " and properties ", propertynames(particles.properties))

function Base.show(io::IO, lagrangian_particles::LagrangianParticles)
    particles = lagrangian_particles.properties
    Tparticle = nameof(eltype(particles))
    properties = propertynames(particles)
    fields = lagrangian_particles.tracked_fields
    Nparticles = length(particles)

    print(io, Nparticles, " LagrangianParticles with eltype ", Tparticle, ":", "\n",
        "├── ", length(properties), " properties: ", properties, "\n",
        "├── particle-wall restitution coefficient: ", lagrangian_particles.restitution, "\n",
        "├── ", length(fields), " tracked fields: ", propertynames(fields), "\n",
        "└── dynamics: ", prettysummary(lagrangian_particles.dynamics, false))
end

# To support interpolation on Flat grids
@inline flattened_node((x, y, z), grid) = (x, y, z)
@inline flattened_node((x, y, z), grid::XFlatGrid) = (y, z)
@inline flattened_node((x, y, z), grid::YFlatGrid) = (x, z)
@inline flattened_node((x, y, z), grid::ZFlatGrid) = (x, y)
@inline flattened_node((x, y, z), grid::YZFlatGrid) = tuple(x)
@inline flattened_node((x, y, z), grid::XZFlatGrid) = tuple(y)
@inline flattened_node((x, y, z), grid::XYFlatGrid) = tuple(z)

include("update_lagrangian_particle_properties.jl")
include("lagrangian_particle_advection.jl")

step_lagrangian_particles!(::Nothing, model, Δt) = nothing

function step_lagrangian_particles!(particles::LagrangianParticles, model, Δt)
    # Update the properties of the Lagrangian particles
    update_lagrangian_particle_properties!(particles, model, Δt)
    
    # Compute dynamics
    particles.dynamics(particles, model, Δt)

    # Advect particles
    advect_lagrangian_particles!(particles, model, Δt)
end

end # module
