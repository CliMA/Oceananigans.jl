
#####
##### Updating particle "field properties"
#####

@kernel function update_property!(particle_property, particles, grid, field, ℓx, ℓy, ℓz)
    p = @index(Global)
    @inbounds begin
        x = particles.x[p]
        y = particles.y[p]
        z = particles.z[p]
        particle_property[p] = interpolate(field, ℓx, ℓy, ℓz, grid, x, y, z)
    end
end

function update_lagrangian_particle_properties!(particles, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    events = []
    workgroup = min(length(particles), 256)
    worksize = length(particles)

    # Update particle "properties"
    for (name, field) in pairs(particles.tracked_fields)
        compute_at!(field, time(model))
        particle_property = getproperty(particles.properties, name)
        ℓx, ℓy, ℓz = map(instantiate, location(field))

        update_property_kernel! = update_property!(device(arch), workgroup, worksize)

        update_event = update_property_kernel!(particle_property,
                                                     particles.properties,
                                                     grid,
                                                     field,
                                                     ℓx, ℓy, ℓz,
                                                     dependencies = device_event(arch))
        push!(events, update_event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

