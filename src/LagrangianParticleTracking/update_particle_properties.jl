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

# Fallback if we skip one cell
@inline adjust_coord(x, args...) where N = x

@inline adjust_coord(x, nodefunc, i, ::Val{0} , grid, rest) = x
@inline adjust_coord(x, nodefunc, i, ::Val{-1}, grid, rest) = nodefunc(Face(), i+1, grid) - (x - nodefunc(Face(), i+1, grid)) * rest
@inline adjust_coord(x, nodefunc, i, ::Val{1} , grid, rest) = nodefunc(Face(), i, grid)   + (nodefunc(Face(), i, grid)   - x) * rest

"""
    pop_immersed_boundary_condition(particles, p, grid, restitution)

If a particle with position `x, y, z` is inside and immersed boundary, correct the 
position based on the previous position (we bounce back a certain restitution from the old cell)
"""
@inline function pop_immersed_particles(particles, p, grid, restitution, old_pos)
    xₚ, yₚ, zₚ = (particles.x[p], particles.y[p], particles.z[p])
    i, j, k   = fractional_indices(xₚ, yₚ, zₚ, (Center(), Center(), Center()), grid.underlying_grid)
    i = Base.unsafe_trunc(Int, i)
    j = Base.unsafe_trunc(Int, j)
    k = Base.unsafe_trunc(Int, k)
   
    if immersed_cell(i, j, k, grid)
        iₒ, jₒ, kₒ   = fractional_indices(old_pos..., (Center(), Center(), Center()), grid.underlying_grid)
        iₒ = Base.unsafe_trunc(Int, iₒ)
        jₒ = Base.unsafe_trunc(Int, jₒ)
        kₒ = Base.unsafe_trunc(Int, kₒ)
       
        if !immersed_cell(iₒ, jₒ, kₒ, grid)
            iᵈ, jᵈ, kᵈ = (i, j, k) .- (iₒ, jₒ, kₒ)
            xₚ = adjust_coord(xₚ, xnode, iₒ, Val(iᵈ), grid)
            yₚ = adjust_coord(yₚ, ynode, jₒ, Val(jᵈ), grid)
            zₚ = adjust_coord(zₚ, znode, kₒ, Val(kᵈ), grid)
        end
    end

    return xₚ, yₚ, zₚ
end

@inline function update_particle_position!(particles, p, restitution, grid::AbstractGrid{FT, TX, TY, TZ}, Δt, velocities) where {FT, TX, TY, TZ}

    # Advect particles using forward Euler.
    @inbounds u = interpolate(velocities.u, Face(), Center(), Center(), grid, particles.x[p], particles.y[p], particles.z[p]) 
    @inbounds v = interpolate(velocities.v, Center(), Face(), Center(), grid, particles.x[p], particles.y[p], particles.z[p]) 
    @inbounds w = interpolate(velocities.w, Center(), Center(), Face(), grid, particles.x[p], particles.y[p], particles.z[p]) 

    j = fractional_y_index(particles.y[p], Center(), grid)
    j = Base.unsafe_trunc(Int, j)

    @inbounds particles.x[p] += calc_correct_velocity_u(u, grid, j) * Δt
    @inbounds particles.y[p] += calc_correct_velocity_v(v, grid) * Δt
    @inbounds particles.z[p] += w * Δt

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

    # old_pos = (particles.x[p], particles.y[p], particles.y[p])

    update_particle_position!(particles, p, restitution, grid.underlying_grid, Δt, velocities) 
    # x, y, z = pop_immersed_particles(particles, p, grid, restitution, old_pos)
    
    # particles.x[p] = x
    # particles.y[p] = y
    # particles.z[p] = z
end

# Linear velocity for RectilinearGrid, Angular velocity for LatitudeLongitudeGrid
@inline calc_correct_velocity_u(U, g::RectilinearGrid, j)       = U
@inline calc_correct_velocity_u(U, g::LatitudeLongitudeGrid, j) = U / (g.radius * hack_cosd(g.φᵃᶜᵃ[j])) * 360 / (2π)

@inline calc_correct_velocity_v(V, g::RectilinearGrid)       = V
@inline calc_correct_velocity_v(V, g::LatitudeLongitudeGrid) = V / g.radius * 360 / (2π)

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
