module LagrangianParticleTracking

export LagrangianParticles, DroguedParticleDynamics, CountingSort, ComparisonSort,
       ParticleBoundaryConditions, ReflectingBoundary, AbsorbingBoundary

using Printf
using Adapt
using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions
using KernelAbstractions: @atomic
using StructArrays
using StructArrays: components

using Oceananigans.Architectures: device, architecture
using Oceananigans.Fields: interpolate, compute!, location
using Oceananigans.Grids
using Oceananigans.Grids: AbstractGrid, hack_cosd,
                          XFlatGrid, YFlatGrid, ZFlatGrid,
                          XYFlatGrid, YZFlatGrid, XZFlatGrid, XYZFlatGrid
using Oceananigans.ImmersedBoundaries
using Oceananigans.ImmersedBoundaries: immersed_cell
using Oceananigans.TimeSteppers: AbstractLagrangianParticles
using Oceananigans.Utils: datatuple, launch!, prettysummary

import Oceananigans: prognostic_state, restore_prognostic_state!
import Oceananigans.TimeSteppers: step_lagrangian_particles!, update_lagrangian_particle_state!
import Oceananigans.OutputWriters: serializeproperty!, fetch_output

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

struct LagrangianParticles{P, B, T, D, Π, S, R} <: AbstractLagrangianParticles
             properties :: P
    boundary_conditions :: B
         tracked_fields :: T
               dynamics :: D
             parameters :: Π
                sorting :: S
              transfers :: R
end

@inline no_dynamics(args...) = nothing

include("particle_species.jl")

"""
    LagrangianParticles(; x, y, z,
                        boundary_conditions = ParticleBoundaryConditions(),
                        dynamics = no_dynamics,
                        parameters = nothing,
                        sorting = nothing)

Construct some `LagrangianParticles` that can be passed to a model. The particles have initial locations
`x`, `y`, and `z`. The behavior of particles at boundaries is specified by `boundary_conditions`;
see [`ParticleBoundaryConditions`](@ref).

`dynamics` is a function of `(lagrangian_particles, model, Δt)` that is called prior to advecting particles.
`parameters` can be accessed inside the `dynamics` function.

`sorting` specifies an algorithm for sorting particles by grid cell, either
[`CountingSort`](@ref) or [`ComparisonSort`](@ref). Default: `nothing` (particles are not sorted).
"""
function LagrangianParticles(; x, y, z,
                             boundary_conditions = ParticleBoundaryConditions(),
                             dynamics = no_dynamics,
                             parameters = nothing,
                             sorting = nothing)
    size(x) == size(y) == size(z) ||
        throw(ArgumentError("x, y, z must all have the same size!"))

    (ndims(x) == 1 && ndims(y) == 1 && ndims(z) == 1) ||
        throw(ArgumentError("x, y, z must have dimension 1 but ndims=($(ndims(x)), $(ndims(y)), $(ndims(z)))"))

    particles = StructArray{Particle}((x, y, z))

    return LagrangianParticles(particles; boundary_conditions, dynamics, parameters, sorting)
end

"""
    LagrangianParticles(properties;
                        boundary_conditions = ParticleBoundaryConditions(),
                        tracked_fields::NamedTuple=NamedTuple(),
                        dynamics = no_dynamics,
                        parameters = nothing,
                        sorting = nothing)

Construct some `LagrangianParticles` that can be passed to a model.

`properties` is either a `StructArray`, whose elements are particles of a single species with
custom properties, or a `NamedTuple` of `StructArray`s, one for each of several particle species,
for example `(; liquid, ice)`. Each species has its own element type and properties, which must
include the positions `x`, `y`, and `z`, stored in arrays of the same type for all species.
Species may be empty and may be resized. `properties` is stored as given, so that
`particles.properties.ice.x` are the `x`-positions of the `ice` particles;
see [`particle_species`](@ref).

Sorting, boundary conditions, and `dynamics` are shared by all species.
Particles may be transferred between species, and removed; see [`transfer_particles!`](@ref).

`boundary_conditions` specifies the behavior of particles at boundaries; see [`ParticleBoundaryConditions`](@ref).
Particles that cross an [`AbsorbingBoundary`](@ref) are removed after they are advected.

A number of `tracked_fields` may be passed in as a `NamedTuple` of fields. Each particle will track the value of each
field. Each tracked field must have a corresponding particle property. So if `T` is a tracked field, then `T` must also
be a custom particle property. With multiple species, `T` is tracked by the species that have the property `T`,
which must include at least one species. Tracked fields are sampled before `dynamics` and refreshed after advection
and boundary handling, so properties match the final particle positions.

`dynamics` is a function of `(lagrangian_particles, model, Δt)` that is called prior to advecting particles.
`parameters` can be accessed inside the `dynamics` function.

`sorting` specifies an algorithm for sorting particles by grid cell, either
[`CountingSort`](@ref) or [`ComparisonSort`](@ref). Default: `nothing` (particles are not sorted).
Sorted particles are reordered at the beginning of every call to `step_lagrangian_particles!`,
so that `dynamics` sees particles stored in order of increasing cell index.
See [`sort_lagrangian_particles!`](@ref).
"""
function LagrangianParticles(properties::Union{StructArray, NamedTuple};
                             boundary_conditions = ParticleBoundaryConditions(),
                             tracked_fields::NamedTuple=NamedTuple(),
                             dynamics = no_dynamics,
                             parameters = nothing,
                             sorting = nothing)

    validate_particle_species(properties)
    validate_tracked_fields(tracked_fields, properties)

    # Scratch space for reordering particle properties, shared by sorting and transfers
    buffers = map_species(species -> similar(species, 0), properties)

    sorting = materialize_sorting(sorting, properties, buffers)
    transfers = materialize_transfers(boundary_conditions, properties, buffers)
    validate_reordering(dynamics, sorting, transfers)

    return LagrangianParticles(properties, boundary_conditions, tracked_fields, dynamics, parameters,
                               sorting, transfers)
end

function validate_tracked_fields(tracked_fields, properties::StructArray)
    for field_name in keys(tracked_fields)
        field_name in propertynames(properties) ||
            throw(ArgumentError("$field_name is a tracked field but $(eltype(properties)) has no $field_name field! " *
                                "You might have to define your own particle type."))
    end
    return nothing
end

function validate_tracked_fields(tracked_fields, properties::NamedTuple)
    for field_name in keys(tracked_fields)
        any(species -> field_name in propertynames(species), properties) ||
            throw(ArgumentError("$field_name is a tracked field but no particle species has a $field_name field! " *
                                "You might have to define your own particle types."))
    end
    return nothing
end

Base.length(lagrangian_particles::LagrangianParticles) = sum(length, particle_species(lagrangian_particles))
Base.size(lagrangian_particles::LagrangianParticles) = (length(lagrangian_particles),)

species_summary(name, species) = string(name, " (", length(species), " ", nameof(eltype(species)), ")")

Base.summary(particles::LagrangianParticles) = particles_summary(particles.properties)

particles_summary(properties::StructArray) =
    string(length(properties), " LagrangianParticles with eltype ", nameof(eltype(properties)),
           " and properties ", propertynames(properties))

particles_summary(properties::NamedTuple) =
    string(sum(length, properties; init=0), " LagrangianParticles in ", length(properties), " species: ",
           join([species_summary(name, species) for (name, species) in pairs(properties)], ", "))

function Base.show(io::IO, lagrangian_particles::LagrangianParticles)
    properties = lagrangian_particles.properties
    fields = lagrangian_particles.tracked_fields

    show_particle_species(io, properties)
    print(io, "├── boundary conditions: ", summary(lagrangian_particles.boundary_conditions), "\n",
              "├── ", length(fields), " tracked fields: ", propertynames(fields), "\n",
              "├── sorting: ", prettysummary(lagrangian_particles.sorting, false), "\n",
              "└── dynamics: ", prettysummary(lagrangian_particles.dynamics, false))
end

function show_particle_species(io, properties::StructArray)
    names = propertynames(properties)
    print(io, length(properties), " LagrangianParticles with eltype ", nameof(eltype(properties)), ":", "\n",
              "├── ", length(names), " properties: ", names, "\n")
end

function show_particle_species(io, properties::NamedTuple)
    print(io, sum(length, properties; init=0), " LagrangianParticles in ", length(properties), " species:", "\n")
    for (name, species) in pairs(properties)
        names = propertynames(species)
        print(io, "├── ", name, ": ", length(species), " ", nameof(eltype(species)),
                  " with ", length(names), " properties ", names, "\n")
    end
end

# To support interpolation on Flat grids
@inline flattened_node((x, y, z), grid) = (x, y, z)
@inline flattened_node((x, y, z), grid::XFlatGrid) = (y, z)
@inline flattened_node((x, y, z), grid::YFlatGrid) = (x, z)
@inline flattened_node((x, y, z), grid::ZFlatGrid) = (x, y)
@inline flattened_node((x, y, z), grid::YZFlatGrid) = tuple(x)
@inline flattened_node((x, y, z), grid::XZFlatGrid) = tuple(y)
@inline flattened_node((x, y, z), grid::XYFlatGrid) = tuple(z)
@inline flattened_node((x, y, z), grid::XYZFlatGrid) = tuple()

include("update_lagrangian_particle_state.jl")
include("particle_boundary_conditions.jl")
include("particle_transfers.jl")
include("lagrangian_particle_advection.jl")
include("drogued_dynamics.jl")
include("lagrangian_particle_sorting.jl")

step_lagrangian_particles!(::Nothing, model, Δt) = nothing

"""
$(TYPEDSIGNATURES)

Advance `particles` by `Δt` with a forward Euler step, using the velocities of `model`.
Particles are sorted (if `particles.sorting` is not `nothing`), `particles.dynamics` is called,
the particles of all species are advected, and absorbed and flagged particles are removed or
transferred between species; see [`transfer_particles!`](@ref).
"""
function step_lagrangian_particles!(particles::LagrangianParticles, model, Δt)
    properties = particles.properties
    resize_transfer_flags!(particles.transfers, properties)
    sort_lagrangian_particles!(particles, model.grid)
    particles.dynamics(particles, model, Δt)
    resize_transfer_flags!(particles.transfers, properties) # `dynamics` may add or remove particles
    advect_lagrangian_particles!(particles, model, Δt)
    transfer_particles!(particles, model.grid)
    return nothing
end

####
#### Extend output writers to support LagrangianParticles
####

serializeproperty!(file, address, p::LagrangianParticles) = serializeproperty!(file, address, p.properties)

fetch_output(lagrangian_particles::LagrangianParticles, model) = fetch_particle_properties(lagrangian_particles.properties)

# A `NamedTuple` of property arrays for a single species, and a `NamedTuple` of species for multiple species
function fetch_particle_properties(properties::StructArray)
    names = propertynames(properties)
    return NamedTuple{names}([getproperty(properties, name) for name in names])
end

fetch_particle_properties(properties::NamedTuple) = map(fetch_particle_properties, properties)

# Checkpointing

function prognostic_state(lagrangian_particles::LagrangianParticles)
    return (; properties = prognostic_state(lagrangian_particles.properties))
end

function restore_prognostic_state!(restored::LagrangianParticles, from)
    restore_particle_properties!(restored.properties, from.properties)

    return restored
end

restore_prognostic_state!(::LagrangianParticles, ::Nothing) = nothing

function restore_particle_properties!(properties::StructArray, from)
    # Particles may have been absorbed or transferred before the checkpoint was written
    resize!(properties, length(from.x))
    restore_prognostic_state!(properties, from)
    return nothing
end

function restore_particle_properties!(properties::NamedTuple, from)
    for name in keys(properties)
        restore_particle_properties!(properties[name], from[name])
    end
    return nothing
end

end # module
