"""
    enforce_boundary_conditions(x, xₗ, xᵣ, ::Type{Bounded})

If a particle with position `x` and domain `xₗ < x < xᵣ` goes through the edge of the domain
along a `Bounded` dimension, put them back at the wall.
"""
@inline function enforce_boundary_conditions(::Type{Bounded}, x, xₗ, xᵣ, restitution)
    x > xᵣ && return xᵣ - (x - xᵣ) * restitution
    x < xₗ && return xₗ + (xₗ - x) * restitution
    return x
end

"""
    enforce_boundary_conditions(x, xₗ, xᵣ, ::Type{Periodic})

If a particle with position `x` and domain `xₗ < x < xᵣ` goes through the edge of the domain
along a `Periodic` dimension, put them on the other side.
"""
@inline function enforce_boundary_conditions(::Type{Periodic}, x, xₗ, xᵣ, restitution)
    x > xᵣ && return xₗ + (x - xᵣ)
    x < xₗ && return xᵣ - (xₗ - x)
    return x
end

@kernel function _advect_particles!(particles, restitution, grid::RegularCartesianGrid{FT, TX, TY, TZ}, Δt, velocities) where {FT, TX, TY, TZ}
    p = @index(Global)

    # Advect particles using forward Euler.
    @inbounds particles.x[p] += interpolate(velocities.u, Face, Cell, Cell, grid, particles.x[p], particles.y[p], particles.z[p]) * Δt
    @inbounds particles.y[p] += interpolate(velocities.v, Cell, Face, Cell, grid, particles.x[p], particles.y[p], particles.z[p]) * Δt
    @inbounds particles.z[p] += interpolate(velocities.w, Cell, Cell, Face, grid, particles.x[p], particles.y[p], particles.z[p]) * Δt

    # Enforce boundary conditions for particles.
    @inbounds particles.x[p] = enforce_boundary_conditions(TX, particles.x[p], grid.xF[1], grid.xF[grid.Nx], restitution)
    @inbounds particles.y[p] = enforce_boundary_conditions(TY, particles.y[p], grid.yF[1], grid.yF[grid.Ny], restitution)
    @inbounds particles.z[p] = enforce_boundary_conditions(TZ, particles.z[p], grid.zF[1], grid.zF[grid.Nz], restitution)
end

advect_particles!(::Nothing, model, Δt) = nothing

function advect_particles!(particles, model, Δt)
    workgroup = min(length(particles), MAX_THREADS_PER_BLOCK)
    worksize = length(particles)
    advect_particles_kernel! = _advect_particles!(device(model.architecture), workgroup, worksize)

    advect_particles_event = advect_particles_kernel!(particles.particles, particles.restitution, model.grid, Δt,
                                                      datatuple(model.velocities),
                                                      dependencies=Event(device(model.architecture)))

    wait(device(model.architecture), advect_particles_event)

    return nothing
end

advect_particles!(model, Δt) = advect_particles!(model.particles, model, Δt)
