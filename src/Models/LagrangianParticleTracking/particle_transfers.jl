#####
##### Transferring particles between species, and removing particles
#####

# Value of `destinations[p]` for particles that are removed, for example by an `AbsorbingBoundary`
const REMOVED_PARTICLE = Int8(-1)

"""
    ParticleTransfers

The `transfers` property of `LagrangianParticles`: a workspace for transferring particles between
species and removing particles, which exists when particles have multiple species or an
[`AbsorbingBoundary`](@ref).

`destinations` mirrors `properties`: for a single species it is a vector, and for multiple species
it is a `NamedTuple` of vectors with one element per particle, where `destinations[p]` is

- `0` (the default), or the index of the species of particle `p`, if particle `p` stays in its species;
- `d` if particle `p` is transferred to species `d` (see [`species_index`](@ref));
- negative if particle `p` is removed. Particles absorbed at an [`AbsorbingBoundary`](@ref)
  are flagged with `-1`.

Particles are transferred and removed after they are advected in every call to `step_lagrangian_particles!`,
and by [`transfer_particles!`](@ref).
Flags are permuted together with the particles when particles are sorted.
"""
struct ParticleTransfers{D, F, C, M, R, B}
    destinations :: D
    flag_buffers :: F
          counts :: C
           masks :: M
           ranks :: R
         buffers :: B
end

function ParticleTransfers(properties, buffers)
    S = length(particle_species(properties))
    x = first(particle_species(properties)).x
    destinations = map_species(species -> fill!(similar(species.x, Int8, length(species)), 0), properties)
    flag_buffers = map_species(species -> similar(species.x, Int8, 0), properties)
    masks = map_species(species -> similar(species.x, Int32, 0), properties)
    ranks = map_species(species -> similar(species.x, Int32, 0), properties)
    counts = similar(x, Int, S * (S + 1))
    return ParticleTransfers(destinations, flag_buffers, counts, masks, ranks, buffers)
end

# Particles of a single species are only removed at absorbing boundaries
materialize_transfers(bcs, properties::StructArray, buffers) = absorbs(bcs) ? ParticleTransfers(properties, buffers) : nothing
materialize_transfers(bcs, properties::NamedTuple, buffers) = ParticleTransfers(properties, buffers)

Base.summary(::ParticleTransfers) = "ParticleTransfers"

transfer_flags(::Nothing) = nothing
transfer_flags(transfers::ParticleTransfers) = transfers.destinations

transfer_flag_buffers(::Nothing) = nothing
transfer_flag_buffers(transfers::ParticleTransfers) = transfers.flag_buffers

#####
##### Flags for absorbed particles
#####

@inline record_absorption!(::Nothing, p, absorbed) = nothing
@inline record_absorption!(flags, p, absorbed) = @inbounds flags[p] = ifelse(absorbed, REMOVED_PARTICLE, flags[p])

resize_transfer_flags!(::Nothing, properties) = nothing

# Flags of particles added since the last particle step are zero
function resize_transfer_flags!(transfers::ParticleTransfers, properties)
    flags = species_tuple(properties, transfers.destinations)
    foreach(flags, particle_species(properties)) do flags, species
        N₀ = length(flags)
        N = length(species)
        if N != N₀
            resize!(flags, N)
            N > N₀ && fill!(view(flags, N₀+1:N), 0)
        end
    end
    return nothing
end

#####
##### Converting particles between species
#####

"""
    convert_particle(destination_type, particle)

Return a particle of species with element type `destination_type`, converted from `particle`,
which is an element of the `StructArray` of another species. The default returns `particle`,
whose properties are then copied by name into the destination species, which works when the
properties of the destination species are a subset of the properties of `particle`.

Extend `convert_particle` to transfer particles to species with properties that `particle` lacks.
The returned object may be a struct or a `NamedTuple` with (at least) the properties of the
destination species. For example,

```julia
using Oceananigans.Models.LagrangianParticleTracking: LagrangianParticleTracking

@inline LagrangianParticleTracking.convert_particle(::Type{<:IceParticle}, p::LiquidParticle) =
    (; x = p.x, y = p.y, z = p.z, radius = p.radius, aspect_ratio = one(p.radius))
```

`convert_particle` is called inside kernels, so it must be type-stable and GPU-compatible.
"""
@inline convert_particle(destination_type, particle) = particle

# Set element `q` of each of `columns` (the components of a `StructArray`) to the property of
# `particle` with the same name. Missing properties are skipped, so that kernels compile for every
# pair of species; conversions are validated by `validate_particle_conversion` before particles are transferred.
@generated function set_particle_properties!(columns::NamedTuple{names}, q, particle) where names
    available = fieldnames(particle)
    assignments = [:(@inbounds columns.$name[q] = getfield(particle, $(QuoteNode(name))))
                   for name in names if name in available]
    return quote
        $(assignments...)
        return nothing
    end
end

function validate_particle_conversion(destination, source)
    particle_type = Base.promote_op(getindex, typeof(source), Int)
    converted_type = Base.promote_op(convert_particle, Type{eltype(destination)}, particle_type)

    if isconcretetype(converted_type)
        missing_properties = filter(name -> !(name in fieldnames(converted_type)), propertynames(destination))
        isempty(missing_properties) ||
            throw(ArgumentError("Cannot transfer particles of type $particle_type to species with element type " *
                                "$(eltype(destination)), because convert_particle returns $converted_type, " *
                                "which lacks the properties $missing_properties. Extend " *
                                "`Oceananigans.Models.LagrangianParticleTracking.convert_particle`."))
    end

    return nothing
end

#####
##### Kernel functions
#####

# The species that particle `p` of species `s` belongs to after transfers, or 0 if it is removed
@inline function transfer_target(d, s, S)
    stays = (d == 0) | (d == s)
    valid = (d > 0) & (d ≤ S)
    return ifelse(stays, s, ifelse(valid, Int(d), 0))
end

# `counts[s + S * (t - 1)]` counts the particles of species `s` that move to species `t`,
# and `counts[s + S * S]` counts the particles of species `s` that are removed.
@inline function count_transfer!((flags, s), p, (counts, S))
    @inbounds t = transfer_target(flags[p], s, S)
    if t != s
        n = s + S * (ifelse(t == 0, S + 1, t) - 1)
        @atomic counts[n] += 1
    end
    return nothing
end

# Write the masks of the groups of particles that end up in each species, for the cumulative sums
# that rank particles within each group. Group `d` occupies column `columns[d]` of `masks`,
# or is inactive if `columns[d] = 0`.
@inline function write_transfer_masks!(masks, p, N, t, columns::Tuple, d)
    column = first(columns)
    if column > 0
        @inbounds masks[p + N * (column - 1)] = t == d
    end
    write_transfer_masks!(masks, p, N, t, Base.tail(columns), d + 1)
    return nothing
end

@inline write_transfer_masks!(masks, p, N, t, ::Tuple{}, d) = nothing

@inline function mask_transfer!((flags, masks, columns, s, N), p, S)
    @inbounds t = transfer_target(flags[p], s, S)
    write_transfer_masks!(masks, p, N, t, columns, 1)
    return nothing
end

transfer_attributes(properties) = values(components(properties))

@inline function write_transferred_particle!(target, q, particle)
    converted = convert_particle(eltype(target), particle)
    set_particle_properties!(components(target), q, converted)
    return nothing
end

# Move particle `p` of species `s` to its position `q` in the species `t` it belongs to after transfers.
# Particles that stay are copied to position `q` = (rank among the particles that stay) of `own_target`.
# Particles transferred to species `t` are converted and appended to the particles that stay in species `t`,
# in order of their source species and then of their index.
@inline function reassign_particle!((source, flags, ranks, columns, N, s, bases, own_target), p, (targets, kept, S))
    @inbounds t = transfer_target(flags[p], s, S)

    if t != 0
        @inbounds rank = ranks[p + N * (columns[t] - 1)]
        if t == s
            copy_elements!(transfer_attributes(own_target), rank, transfer_attributes(source), p)
        else
            @inbounds q = kept[t] + bases[t] + rank
            @inbounds particle = source[p]
            apply_at_index(write_transferred_particle!, t, targets, q, particle)
        end
    end

    return nothing
end

@inline zero_flag!((flags,), p, shared_args) = @inbounds flags[p] = 0

#####
##### Transferring particles
#####

# Number consecutive active groups, with 0 for inactive groups
function group_columns(active)
    column = 0
    return map(a -> a ? (column += 1) : 0, active)
end

transfer_particles!(particles, grid) = nothing

"""
$(TYPEDSIGNATURES)

Transfer `particles` between species and remove particles, as flagged in `particles.transfers.destinations`
(see [`ParticleTransfers`](@ref)), and reset the flags to zero. Particles that cross an [`AbsorbingBoundary`](@ref)
are flagged for removal. Called by `step_lagrangian_particles!` after particles are advected.

Particles transferred to species `d` are converted with [`convert_particle`](@ref) and appended to species `d`
in order of their source species and then of their index. The remaining particles of each species keep their
order. The numbers of particles that leave each species are copied to the host once.
"""
function transfer_particles!(particles::LagrangianParticles, grid)
    transfers = particles.transfers
    isnothing(transfers) && return nothing

    properties = particles.properties
    species = particle_species(properties)
    counts = map(length, species)
    sum(counts) == 0 && return nothing

    S = length(species)
    numbers = species_numbers(counts)
    arch = architecture(grid)
    flags = species_tuple(properties, transfers.destinations)

    foreach(flags, species) do flags, species
        length(flags) == length(species) ||
            throw(ArgumentError("The transfer flags of a species have length $(length(flags)) " *
                                "but the species has $(length(species)) particles."))
    end

    # Count the particles that leave each species, by destination
    fill!(transfers.counts, 0)
    launch_species!(arch, count_transfer!, counts, map(tuple, flags, numbers), (transfers.counts, S))
    transfer_counts = reshape(Array(transfers.counts), S, S + 1)

    leaving = map(s -> sum(view(transfer_counts, s, :)), numbers)
    all(iszero, leaving) && return nothing

    incoming = map(d -> sum(view(transfer_counts, :, d)), numbers)
    kept = map(-, counts, leaving)
    new_counts = map(+, kept, incoming)
    bases = map(s -> map(d -> sum(view(transfer_counts, 1:s-1, d)), numbers), numbers)

    for s in 1:S, d in 1:S
        if d != s && transfer_counts[s, d] > 0
            validate_particle_conversion(species[d], species[s])
        end
    end

    # Rank the particles of each group (the particles that end up in each species) of each source species
    # with cumulative sums of the group masks. Only species that lose particles are processed.
    columns = map(s -> group_columns(map(d -> d == s ? kept[s] > 0 : transfer_counts[s, d] > 0, numbers)), numbers)
    source_counts = map((N, L) -> L > 0 ? N : 0, counts, leaving)
    masks = species_tuple(properties, transfers.masks)
    ranks = species_tuple(properties, transfers.ranks)

    foreach(masks, ranks, source_counts, columns) do masks, ranks, N, columns
        resize!(masks, N * maximum(columns))
        resize!(ranks, N * maximum(columns))
    end

    launch_species!(arch, mask_transfer!, source_counts, map(tuple, flags, masks, columns, numbers, counts), S)

    foreach(masks, ranks, source_counts, columns) do masks, ranks, N, columns
        for column in 1:maximum(columns)
            n = (column - 1) * N + 1 : column * N
            cumsum!(view(ranks, n), view(masks, n))
        end
    end

    # Species that lose particles are compacted into buffers; other species receive particles in place.
    buffers = species_tuple(properties, transfers.buffers)
    compacted = map(L -> L > 0, leaving)
    targets = map((compacted, species, buffer) -> compacted ? buffer : species, compacted, species, buffers)
    foreach(resize!, targets, new_counts)

    reassign_args = map(tuple, species, flags, ranks, columns, counts, numbers, bases, targets)
    launch_species!(arch, reassign_particle!, source_counts, reassign_args, (targets, kept, S))

    # Copy compacted species back from their buffers
    foreach(compacted, species, new_counts) do compacted, species, N
        compacted && resize!(species, N)
    end

    copy_counts = map((compacted, N) -> compacted ? N : 0, compacted, new_counts)
    copy_args = map((species, buffer) -> (transfer_attributes(species), transfer_attributes(buffer)), species, buffers)
    launch_species!(arch, copy_particle!, copy_counts, copy_args, nothing)

    # Reset the flags
    foreach(resize!, flags, new_counts)
    launch_species!(arch, zero_flag!, new_counts, map(tuple, flags), nothing)

    return nothing
end
