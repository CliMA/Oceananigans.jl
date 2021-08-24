"""
    enforce_boundary_conditions(x, xₗ, xᵣ, ::Bounded)

If a particle with position `x` and domain `xₗ < x < xᵣ` goes through the edge of the domain
along a `Bounded` dimension, put them back at the wall.
"""
@inline function enforce_boundary_conditions(::Bounded, x, xₗ, xᵣ, restitution)
    x > xᵣ && return xᵣ - (x - xᵣ) * restitution
    x < xₗ && return xₗ + (xₗ - x) * restitution
    return x
end

"""
    enforce_boundary_conditions(x, xₗ, xᵣ, ::Periodic)

If a particle with position `x` and domain `xₗ < x < xᵣ` goes through the edge of the domain
along a `Periodic` dimension, put them on the other side.
"""
@inline function enforce_boundary_conditions(::Periodic, x, xₗ, xᵣ, restitution)
    x > xᵣ && return xₗ + (x - xᵣ)
    x < xₗ && return xᵣ - (xₗ - x)
    return x
end

@kernel function _advect_particles!(particles, restitution, grid::RegularRectilinearGrid{FT, TX, TY, TZ}, Δt, velocities) where {FT, TX, TY, TZ}
    p = @index(Global)

    # Advect particles using forward Euler.
    @inbounds particles.x[p] += interpolate(velocities.u, Face(), Center(), Center(), grid, particles.x[p], particles.y[p], particles.z[p]) * Δt
    @inbounds particles.y[p] += interpolate(velocities.v, Center(), Face(), Center(), grid, particles.x[p], particles.y[p], particles.z[p]) * Δt
    @inbounds particles.z[p] += interpolate(velocities.w, Center(), Center(), Face(), grid, particles.x[p], particles.y[p], particles.z[p]) * Δt

    # Enforce boundary conditions for particles.
    @inbounds particles.x[p] = enforce_boundary_conditions(TX(), particles.x[p], grid.xF[1], grid.xF[grid.Nx], restitution)
    @inbounds particles.y[p] = enforce_boundary_conditions(TY(), particles.y[p], grid.yF[1], grid.yF[grid.Ny], restitution)
    @inbounds particles.z[p] = enforce_boundary_conditions(TZ(), particles.z[p], grid.zF[1], grid.zF[grid.Nz], restitution)
end

@kernel function update_field_property!(particle_property, particles, grid, field, LX, LY, LZ)
    p = @index(Global)

    @inbounds particle_property[p] = interpolate(field, LX, LY, LZ, grid, particles.x[p], particles.y[p], particles.z[p])
end

function update_particle_properties!(lagrangian_particles, model, Δt)

    # Update tracked field properties.

    workgroup = min(length(lagrangian_particles), MAX_GPU_THREADS_PER_BLOCK)
    worksize = length(lagrangian_particles)

    events = []

    for (field_name, tracked_field) in pairs(lagrangian_particles.tracked_fields)
        compute!(tracked_field)
        particle_property = getproperty(lagrangian_particles.properties, field_name)
        LX, LY, LZ = location(tracked_field)

        update_field_property_kernel! = update_field_property!(device(model.architecture), workgroup, worksize)

        update_event = update_field_property_kernel!(particle_property, lagrangian_particles.properties, model.grid,
                                                     datatuple(tracked_field), LX(), LY(), LZ(),
                                                     dependencies=Event(device(model.architecture)))
        push!(events, update_event)
    end

    wait(device(model.architecture), MultiEvent(Tuple(events)))

    # Compute dynamics

    lagrangian_particles.dynamics(lagrangian_particles, model, Δt)

    # Advect particles

    advect_particles_kernel! = _advect_particles!(device(model.architecture), workgroup, worksize)

    advect_particles_event = advect_particles_kernel!(lagrangian_particles.properties, lagrangian_particles.restitution, model.grid, Δt,
                                                      datatuple(model.velocities),
                                                      dependencies=Event(device(model.architecture)))

    wait(device(model.architecture), advect_particles_event)

    return nothing
end

update_particle_properties!(::Nothing, model, Δt) = nothing

update_particle_properties!(model, Δt) = update_particle_properties!(model.particles, model, Δt)
