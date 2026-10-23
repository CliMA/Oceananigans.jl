#####
##### Particle species
#####
##### `LagrangianParticles` store either a single species of particles in a `StructArray`, or several
##### species in a `NamedTuple` of `StructArray`s. All operations go through `particle_species`, which
##### returns a tuple of species, so that a single species is the special case of a 1-tuple.
##### Per-species workspaces (for example, transfer flags and sorting permutations) mirror the
##### structure of `properties`: a single object for a single species, or a `NamedTuple` with the same
##### keys as `properties` for multiple species.
#####

"""
$(TYPEDSIGNATURES)

Return a tuple of the particle species in `properties`: `(properties,)` for a single species stored
in a `StructArray`, or the values of a `NamedTuple` of `StructArray`s.

```jldoctest
julia> using Oceananigans, StructArrays

julia> using Oceananigans.Models.LagrangianParticleTracking: particle_species

julia> liquid = StructArray((x=zeros(2), y=zeros(2), z=zeros(2), radius=ones(2)));

julia> ice = StructArray((x=zeros(3), y=zeros(3), z=zeros(3), radius=ones(3), aspect_ratio=ones(3)));

julia> map(length, particle_species((; liquid, ice)))
(2, 3)

julia> map(length, particle_species(liquid))
(2,)
```
"""
particle_species(properties::StructArray) = tuple(properties)
particle_species(properties::NamedTuple) = values(properties)
particle_species(particles::LagrangianParticles) = particle_species(particles.properties)

"""
$(TYPEDSIGNATURES)

Return the index of the species `name` of `particles`, for flagging particle transfers to
that species; see [`transfer_particles!`](@ref).
"""
function species_index(particles::LagrangianParticles, name::Symbol)
    names = species_names(particles.properties)
    s = findfirst(==(name), names)
    isnothing(s) && throw(ArgumentError("LagrangianParticles have no species $name; species are $names."))
    return s
end

species_names(properties::StructArray) = ()
species_names(properties::NamedTuple) = keys(properties)

# Per-species values of `container`, whose structure mirrors `properties`
species_tuple(::StructArray, container) = tuple(container)
species_tuple(::StructArray, ::Nothing) = tuple(nothing)
species_tuple(::NamedTuple, container::NamedTuple) = values(container)
species_tuple(properties::NamedTuple, ::Nothing) = map(species -> nothing, values(properties))

# An object with the structure of `properties` that holds the per-species `values`
mirror_species(::StructArray, values::Tuple{Any}) = first(values)
mirror_species(::NamedTuple{names}, values::Tuple) where names = NamedTuple{names}(values)

map_species(f, properties) = mirror_species(properties, map(f, particle_species(properties)))

function validate_particle_species(properties::StructArray)
    for name in (:x, :y, :z)
        name in propertynames(properties) ||
            throw(ArgumentError("Particles with eltype $(eltype(properties)) have no $name property!"))
    end
    return nothing
end

function validate_particle_species(properties::NamedTuple)
    isempty(properties) && throw(ArgumentError("At least one particle species must be provided."))

    for (name, species) in pairs(properties)
        species isa StructArray ||
            throw(ArgumentError("Particle species $name must be a StructArray, but is a $(typeof(species))."))
        validate_particle_species(species)
    end

    # Positions are stored in the same type of array for all species
    position_types = map(species -> typeof(species.x), values(properties))
    all(==(first(position_types)), position_types) ||
        throw(ArgumentError("The positions of all particle species must have the same type, but have types $position_types."))

    return nothing
end

#####
##### Launching kernels over the particles of all species
#####

"""
    SuperrangeLaunch()

Launch one kernel over the particles of all species. Species `s` occupies the contiguous
block of indices `O + 1 : O + Nₛ` of the global index range `1 : sum(Nₛ)`, where `O` is
the number of particles in species `1 : s - 1`. Each thread selects its species with an unrolled,
type-stable recursion over the tuple of species, so that warps execute code specialized for
a single element type except at block boundaries.
"""
struct SuperrangeLaunch end

"""
    PerSpeciesLaunch()

Launch one kernel for each species.
"""
struct PerSpeciesLaunch end

"""
    species_launch_strategy()

Return the default strategy for launching kernels over the particles of all species,
which is [`SuperrangeLaunch`](@ref). Redefine this method to use [`PerSpeciesLaunch`](@ref).

On an A100, advecting 10⁴–10⁵ particles in 2–3 species is 1.4–1.7 times faster with superrange
launches than with per-species launches, because kernel launch overhead dominates; for 10⁶–10⁷
particles both strategies are equally fast.
"""
species_launch_strategy() = SuperrangeLaunch()

const particle_workgroup = 256

# Particle kernels are launched with a dynamic `ndrange`, so that changing the number of particles
# does not trigger the compilation of new kernels.
function launch_particle_kernel!(arch, kernel!, N, args...)
    N > 0 || return nothing
    loop! = kernel!(device(arch), particle_workgroup)
    loop!(args...; ndrange=N)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Apply `op(species_args[s], p, shared_args)` to particle `p` of every species `s`, where species `s`
has `counts[s]` particles. `species_args` and `counts` are tuples with one element per species.
Species with `counts[s] = 0` are skipped. The kernels are launched with `strategy`;
see [`SuperrangeLaunch`](@ref) and [`PerSpeciesLaunch`](@ref).
"""
launch_species!(arch, op, counts, species_args, shared_args, strategy=species_launch_strategy()) =
    launch_species!(strategy, arch, op, counts, species_args, shared_args)

launch_species!(::SuperrangeLaunch, arch, op, counts, species_args, shared_args) =
    launch_particle_kernel!(arch, _species_superrange!, sum(counts), op, counts, species_args, shared_args)

function launch_species!(::PerSpeciesLaunch, arch, op, counts, species_args, shared_args)
    foreach(counts, species_args) do N, args
        launch_particle_kernel!(arch, _single_species!, N, op, args, shared_args)
    end
    return nothing
end

@kernel function _species_superrange!(op, counts, species_args, shared_args)
    g = @index(Global, Linear)
    apply_to_species(op, g, counts, species_args, shared_args)
end

@kernel function _single_species!(op, args, shared_args)
    p = @index(Global, Linear)
    op(args, p, shared_args)
end

"""
$(TYPEDSIGNATURES)

Apply `op` to the particle with global index `g` in the superrange of all species, where species `s`
has `counts[s]` particles: `op(species_args[s], p, shared_args)`, where `p` is the index of the particle
within species `s`.
"""
@inline function apply_to_species(op, g, counts::Tuple, species_args::Tuple, shared_args)
    N = first(counts)
    if g ≤ N
        op(first(species_args), g, shared_args)
    else
        apply_to_species(op, g - N, Base.tail(counts), Base.tail(species_args), shared_args)
    end
    return nothing
end

@inline apply_to_species(op, g, ::Tuple{}, ::Tuple{}, shared_args) = nothing

# Apply `f(items[d], args...)` for a runtime index `d`, with an unrolled and type-stable recursion
@inline function apply_at_index(f, d, items::Tuple, args...)
    if d == 1
        f(first(items), args...)
    else
        apply_at_index(f, d - 1, Base.tail(items), args...)
    end
    return nothing
end

@inline apply_at_index(f, d, ::Tuple{}, args...) = nothing

#####
##### Particle attributes: the per-particle arrays that are reordered together
#####

flag_arrays(::Nothing) = ()
flag_arrays(flags::AbstractArray) = tuple(flags)

# The properties and transfer flags of a species
particle_attributes(properties, flags) = (values(components(properties))..., flag_arrays(flags)...)

# Copy element `q` of each array in `sources` to element `p` of the corresponding array in `destinations`
@inline function copy_elements!(destinations::Tuple, p, sources::Tuple, q)
    @inbounds first(destinations)[p] = first(sources)[q]
    copy_elements!(Base.tail(destinations), p, Base.tail(sources), q)
    return nothing
end

@inline copy_elements!(::Tuple{}, p, ::Tuple{}, q) = nothing

@inline function gather_particle!((permutation, buffers, attributes), p, shared_args)
    @inbounds q = permutation[p]
    copy_elements!(buffers, p, attributes, q)
    return nothing
end

@inline copy_particle!((destinations, sources), p, shared_args) = copy_elements!(destinations, p, sources, p)
