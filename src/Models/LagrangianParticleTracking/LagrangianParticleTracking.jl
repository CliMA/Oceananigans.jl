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
using Oceananigans.TimeSteppers: AbstractLagrangianParticles, QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper
using Oceananigans.Utils: datatuple, launch!, prettysummary

import Oceananigans: prognostic_state, restore_prognostic_state!
import Oceananigans.TimeSteppers: step_lagrangian_particles!, substep_lagrangian_particles!,
                                  update_lagrangian_particle_state!, materialize_lagrangian_particles
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

struct LagrangianParticles{P, B, T, D, Π, S, R, M} <: AbstractLagrangianParticles
             properties :: P
    boundary_conditions :: B
         tracked_fields :: T
               dynamics :: D
             parameters :: Π
                sorting :: S
                removal :: R
            timestepper :: M
end

@inline no_dynamics(args...) = nothing

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
    LagrangianParticles(particles::StructArray;
                        boundary_conditions = ParticleBoundaryConditions(),
                        tracked_fields::NamedTuple=NamedTuple(),
                        dynamics = no_dynamics,
                        parameters = nothing,
                        sorting = nothing)

Construct some `LagrangianParticles` that can be passed to a model. The `particles` should be a `StructArray`
and can contain custom fields.

`boundary_conditions` specifies the behavior of particles at boundaries; see [`ParticleBoundaryConditions`](@ref).
Particles that cross an [`AbsorbingBoundary`](@ref) are removed after they are advected.

A number of `tracked_fields` may be passed in as a `NamedTuple` of fields. Each particle will track the value of each
field. Each tracked field must have a corresponding particle property. So if `T` is a tracked field, then `T` must also
be a custom particle property. Tracked fields are sampled before `dynamics` and refreshed after advection
and boundary handling, so properties match the final particle positions.

`dynamics` is a function of `(lagrangian_particles, model, Δt)` that is called once per time step,
at the beginning of the first stage and prior to advecting particles, with the full time step `Δt`.
`parameters` can be accessed inside the `dynamics` function.

`sorting` specifies an algorithm for sorting particles by grid cell, either
[`CountingSort`](@ref) or [`ComparisonSort`](@ref). Default: `nothing` (particles are not sorted).
Sorted particles are reordered once per time step, at the beginning of the first stage,
so that `dynamics` sees particles stored in order of increasing cell index.
See [`sort_lagrangian_particles!`](@ref).

Particles are advanced with the time-stepping scheme of the model they belong to;
see [`substep_lagrangian_particles!`](@ref).
"""
function LagrangianParticles(particles::StructArray;
                             boundary_conditions = ParticleBoundaryConditions(),
                             tracked_fields::NamedTuple=NamedTuple(),
                             dynamics = no_dynamics,
                             parameters = nothing,
                             sorting = nothing)

    for (field_name, tracked_field) in pairs(tracked_fields)
        field_name in propertynames(particles) ||
            throw(ArgumentError("$field_name is a tracked field but $(eltype(particles)) has no $field_name field! " *
                                "You might have to define your own particle type."))
    end

    sorting = materialize_sorting(sorting, particles)
    removal = materialize_removal(boundary_conditions, particles)
    validate_reordering(dynamics, sorting, removal)

    timestepper = nothing # materialized by the model constructor; see `materialize_lagrangian_particles`

    return LagrangianParticles(particles, boundary_conditions, tracked_fields, dynamics, parameters,
                               sorting, removal, timestepper)
end

Base.size(lagrangian_particles::LagrangianParticles) = size(lagrangian_particles.properties)
Base.length(lagrangian_particles::LagrangianParticles) = length(lagrangian_particles.properties)

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
        "├── boundary conditions: ", summary(lagrangian_particles.boundary_conditions), "\n",
        "├── ", length(fields), " tracked fields: ", propertynames(fields), "\n",
        "├── sorting: ", prettysummary(lagrangian_particles.sorting, false), "\n",
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
@inline flattened_node((x, y, z), grid::XYZFlatGrid) = tuple()

include("update_lagrangian_particle_state.jl")
include("particle_boundary_conditions.jl")
include("lagrangian_particle_advection.jl")
include("drogued_dynamics.jl")
include("lagrangian_particle_sorting.jl")
include("lagrangian_particle_time_stepping.jl")

####
#### Extend output writers to support LagrangianParticles
####

serializeproperty!(file, address, p::LagrangianParticles) = serializeproperty!(file, address, p.properties)

function fetch_output(lagrangian_particles::LagrangianParticles, model)
    particle_properties = lagrangian_particles.properties
    names = propertynames(particle_properties)
    return NamedTuple{names}([getproperty(particle_properties, name) for name in names])
end

# Checkpointing

function prognostic_state(lagrangian_particles::LagrangianParticles)
    return (; properties = prognostic_state(lagrangian_particles.properties),
              timestepper = prognostic_state(lagrangian_particles.timestepper))
end

function restore_prognostic_state!(restored::LagrangianParticles, from)
    # Particles may have been absorbed before the checkpoint was written
    resize!(restored.properties, length(from.properties.x))
    restore_prognostic_state!(restored.properties, from.properties)

    if hasproperty(from, :timestepper)
        restore_prognostic_state!(restored.timestepper, from.timestepper)
    end

    return restored
end

restore_prognostic_state!(::LagrangianParticles, ::Nothing) = nothing

end # module
