
#####
##### Updating particle "field properties"
#####

@kernel function update_property!(particle_property, particles, grid, field, ℓx, ℓy, ℓz)
    p = @index(Global)
    @inbounds begin
        x = particles.x[p]
        y = particles.y[p]
        z = particles.z[p]
        X = flattened_node((x, y, z), grid)
        particle_property[p] = interpolate(X, field, (ℓx, ℓy, ℓz), grid)
    end
end

function update_lagrangian_particle_properties!(particles, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    workgroup = min(length(particles), 256)
    worksize = length(particles)

    # Update particle "properties"
    for (name, field) in pairs(particles.tracked_fields)
        compute!(field)
        particle_property = getproperty(particles.properties, name)
        ℓx, ℓy, ℓz = map(instantiate, location(field))

        update_field_property_kernel! = update_property!(device(arch), workgroup, worksize)

        update_field_property_kernel!(particle_property, particles.properties, model.grid,
                                      datatuple(field), ℓx, ℓy, ℓz)
    end

    return nothing
end

