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

@inline positive_x_immersed_boundary(i, j, k, grid) = !inactive_cell(i, j, k, grid) & inactive_cell(i-1, j, k, grid)
@inline positive_y_immersed_boundary(i, j, k, grid) = !inactive_cell(i, j, k, grid) & inactive_cell(i, j-1, k, grid)
@inline positive_z_immersed_boundary(i, j, k, grid) = !inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)

@inline negative_x_immersed_boundary(i, j, k, grid) = !inactive_cell(i, j, k, grid) & inactive_cell(i+1, j, k, grid)
@inline negative_y_immersed_boundary(i, j, k, grid) = !inactive_cell(i, j, k, grid) & inactive_cell(i, j+1, k, grid)
@inline negative_z_immersed_boundary(i, j, k, grid) = !inactive_cell(i, j, k, grid) & inactive_cell(i, j, k+1, grid)

@inline positive_correction(xₚ, xᶠ, restitution, func, i, j, k, grid) = func(i, j, k, grid) && xₚ < xᶠ ? xᶠ + (xᶠ - xₚ) * restitution : xₚ
@inline negative_correction(xₚ, xᶠ, restitution, func, i, j, k, grid) = func(i, j, k, grid) && xₚ > xᶠ ? xᶠ - (xₚ - xᶠ) * restitution : xₚ

"""
    enforce_immersed_boundary_condition(particles, p, grid, restitution)

If a particle with position `x, y, z` is at the edge of an immersed boundary, correct the 
position as to avoid 
"""
@inline function enforce_immersed_boundary_condition(particles, p, grid, restitution)
    xₚ, yₚ, zₚ = (particles.x[p], particles.y[p], particles.z[p])
    i, j, k   = fractional_indices(xₚ, yₚ, zₚ, (Center(), Center(), Center()), grid.underlying_grid)
    i = Base.unsafe_trunc(Int, i)
    j = Base.unsafe_trunc(Int, j)
    k = Base.unsafe_trunc(Int, k)
    
    xᶠ⁻ = xnode(Face(), i, grid)
    yᶠ⁻ = ynode(Face(), j, grid)
    zᶠ⁻ = znode(Face(), k, grid)
    xᶠ⁺ = xnode(Face(), i + 1, grid)
    yᶠ⁺ = ynode(Face(), j + 1, grid)
    zᶠ⁺ = znode(Face(), k + 1, grid)

    xₚ = positive_correction(xₚ, xᶠ⁻, restitution, positive_x_immersed_boundary, i, j, k, grid)
    yₚ = positive_correction(yₚ, yᶠ⁻, restitution, positive_y_immersed_boundary, i, j, k, grid)
    zₚ = positive_correction(zₚ, zᶠ⁻, restitution, positive_z_immersed_boundary, i, j, k, grid)
    
    xₚ = negative_correction(xₚ, xᶠ⁺, restitution, negative_x_immersed_boundary, i, j, k, grid)
    yₚ = negative_correction(yₚ, yᶠ⁺, restitution, negative_y_immersed_boundary, i, j, k, grid)
    yₚ = negative_correction(zₚ, zᶠ⁺, restitution, negative_z_immersed_boundary, i, j, k, grid)

    return xₚ, yₚ, zₚ
end

@inline function update_particle_position!(particles, p, restitution, grid::AbstractGrid{FT, TX, TY, TZ}, Δt, velocities) where {FT, TX, TY, TZ}

    # Advect particles using forward Euler.
    @inbounds u = interpolate(velocities.u, Face(), Center(), Center(), grid, particles.x[p], particles.y[p], particles.z[p]) 
    @inbounds v = interpolate(velocities.v, Center(), Face(), Center(), grid, particles.x[p], particles.y[p], particles.z[p]) 
    @inbounds w = interpolate(velocities.w, Center(), Center(), Face(), grid, particles.x[p], particles.y[p], particles.z[p]) 

    @inbounds particles.x[p] += calc_correct_velocity(u, grid) * Δt
    @inbounds particles.y[p] += calc_correct_velocity(v, grid) * Δt
    @inbounds particles.z[p] += calc_correct_velocity(w, grid) * Δt

    x, y, z = return_face_metrics(grid)

    # Enforce boundary conditions for particles.
    @inbounds particles.x[p] = enforce_boundary_conditions(TX(), particles.x[p], x[1], x[grid.Nx], restitution)
    @inbounds particles.y[p] = enforce_boundary_conditions(TY(), particles.y[p], y[1], y[grid.Ny], restitution)
    @inbounds particles.z[p] = enforce_boundary_conditions(TZ(), particles.z[p], z[1], z[grid.Nz], restitution)
end

@kernel function _advect_particles!(particles, restitution, grid::AbstractUnderlyingGrid, Δt, velocities) 
    p = @index(Global)
    update_particle_position!(particles, p, restitution, grid, Δt, velocities) 
end

@kernel function _advect_particles!(particles, restitution, grid::ImmersedBoundaryGrid, Δt, velocities)
    p = @index(Global)
    update_particle_position!(particles, p, restitution, grid.underlying_grid, Δt, velocities) 
    @inbounds particles.x[p], particles.y[p], particles.z[p] = enforce_immersed_boundary_condition(particles, p, grid, restitution)
end

# Linear velocity for RectilinearGrid, Angular velocity for LatitudeLongitudeGrid
@inline calc_correct_velocity(U, g::RectilinearGrid)       = U
@inline calc_correct_velocity(U, g::LatitudeLongitudeGrid) = U / g.radius

@inline return_face_metrics(g::LatitudeLongitudeGrid) = (g.λᶠᵃᵃ, g.φᵃᶠᵃ, g.zᵃᵃᶠ)
@inline return_face_metrics(g::RectilinearGrid)       = (g.xᶠᵃᵃ, g.yᵃᶠᵃ, g.zᵃᵃᶠ)
@inline return_face_metrics(g::ImmersedBoundaryGrid)  = return_face_metrics(g.underlying_grid)

@kernel function update_field_property!(particle_property, particles, grid, field, LX, LY, LZ)
    p = @index(Global)

    @inbounds particle_property[p] = interpolate(field, LX, LY, LZ, grid, particles.x[p], particles.y[p], particles.z[p])
end

function update_particle_properties!(lagrangian_particles, model, Δt)

    # Update tracked field properties.
    workgroup = min(length(lagrangian_particles), 256)
    worksize = length(lagrangian_particles)

    arch = architecture(model.grid)

    events = []

    for (field_name, tracked_field) in pairs(lagrangian_particles.tracked_fields)
        compute!(tracked_field)
        particle_property = getproperty(lagrangian_particles.properties, field_name)
        LX, LY, LZ = location(tracked_field)

        update_field_property_kernel! = update_field_property!(device(arch), workgroup, worksize)

        update_event = update_field_property_kernel!(particle_property, lagrangian_particles.properties, model.grid,
                                                     datatuple(tracked_field), LX(), LY(), LZ(),
                                                     dependencies=Event(device(arch)))
        push!(events, update_event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    # Compute dynamics

    lagrangian_particles.dynamics(lagrangian_particles, model, Δt)

    # Advect particles



    advect_particles_kernel! = _advect_particles!(device(arch), workgroup, worksize)

    advect_particles_event = advect_particles_kernel!(lagrangian_particles.properties, lagrangian_particles.restitution, model.grid, Δt,
                                                      datatuple(model.velocities),
                                                      dependencies=Event(device(arch)))

    wait(device(arch), advect_particles_event)

    return nothing
end

update_particle_properties!(::Nothing, model, Δt) = nothing

update_particle_properties!(model, Δt) = update_particle_properties!(model.particles, model, Δt)
