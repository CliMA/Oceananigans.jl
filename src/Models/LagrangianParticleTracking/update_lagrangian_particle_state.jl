#####
##### Updating particle "field properties"
#####

@inline function update_tracked_property!((particle_property, particles), p, (grid, field, ℓx, ℓy, ℓz))
    @inbounds begin
        x = particles.x[p]
        y = particles.y[p]
        z = particles.z[p]
        X = flattened_node((x, y, z), grid)
        particle_property[p] = interpolate(X, field, (ℓx, ℓy, ℓz), grid)
    end
    return nothing
end

# Species without the tracked property are skipped
@inline update_tracked_property!(::Tuple{Nothing, Any}, p, shared_args) = nothing

tracked_property(species, name) = name in propertynames(species) ? getproperty(species, name) : nothing
tracking_count(species, name) = name in propertynames(species) ? length(species) : 0

update_lagrangian_particle_state!(particles, model) = nothing

function update_lagrangian_particle_state!(particles::LagrangianParticles, model)
    grid = model.grid
    arch = architecture(grid)
    species = particle_species(particles)

    # Update particle "properties" of the species that have them
    for (name, field) in pairs(particles.tracked_fields)
        compute!(field)
        ℓx, ℓy, ℓz = map(instantiate, location(field))
        counts = map(species -> tracking_count(species, name), species)
        species_args = map(species -> (tracked_property(species, name), species), species)
        launch_species!(arch, update_tracked_property!, counts, species_args, (grid, datatuple(field), ℓx, ℓy, ℓz))
    end

    return nothing
end
