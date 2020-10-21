module LagrangianParticleTracking

export LagrangianParticles, advect_particles!

using KernelAbstractions

using Oceananigans.Grids

using Oceananigans.Architectures: device
using Oceananigans.Fields: interpolate, datatuple
using Oceananigans.Utils: MAX_THREADS_PER_BLOCK

import Base: size, length

struct LagrangianParticles{X, Y, Z}
    x :: X
    y :: Y
    z :: Z
end

function LagrangianParticles(; x, y, z)
    size(x) == size(y) == size(z) ||
        error("x, y, z must all have the same size!")
        
    (ndims(x) == 1 && ndims(y) == 1 && ndims(z) == 1) ||
        error("x, y, z must have dimension 1 but ndims=($(ndims(x)), $(ndims(y)), $(ndims(z)))")

    return LagrangianParticles(x, y, z)
end

size(particles::LagrangianParticles) = size(particles.x)
length(particles::LagrangianParticles) = length(particles.x)

@inline enforce_periodicity(x, xₗ, xᵣ, ::Type{Bounded})  = x

@inline function enforce_periodicity(x, xₗ, xᵣ, ::Type{Periodic})
    x > xᵣ && return xₗ + (x - xᵣ)
    x < xₗ && return xᵣ - (xₗ - x)
    return x
end

@kernel function _advect_particles!(particles, grid::RegularCartesianGrid{FT, TX, TY, TZ}, Δt, velocities) where {FT, TX, TY, TZ}
    p = @index(Global)

    x, y, z = particles.x[p], particles.y[p], particles.z[p]

    ℑu = interpolate(velocities.u, x, y, z)
    ℑv = interpolate(velocities.v, x, y, z)
    ℑw = interpolate(velocities.w, x, y, z)

    # Advect!
    @inbounds particles.x[p] += ℑu * Δt
    @inbounds particles.y[p] += ℑv * Δt
    @inbounds particles.z[p] += ℑw * Δt

    # If particle go out of the domain along a periodic dimension, put them on the other side.
    @inbounds particles.x[p] = enforce_periodicity(particles.x[p], grid.xF[1], grid.xF[grid.Nx], TX)
    @inbounds particles.y[p] = enforce_periodicity(particles.y[p], grid.yF[1], grid.yF[grid.Ny], TY)
    @inbounds particles.z[p] = enforce_periodicity(particles.z[p], grid.zF[1], grid.zF[grid.Nz], TZ)

    # If particles go through a wall, put them back at the wall.
    @inbounds particles.x[p] = clamp(particles.x[p], grid.xF[1], grid.xF[grid.Nx])
    @inbounds particles.y[p] = clamp(particles.y[p], grid.yF[1], grid.yF[grid.Ny])
    @inbounds particles.z[p] = clamp(particles.z[p], grid.zF[1], grid.zF[grid.Nz])
end

function advect_particles!(model, Δt)
    workgroup = min(length(model.particles), MAX_THREADS_PER_BLOCK)
    worksize = length(model.particles)
    advect_particles_kernel! = _advect_particles!(device(model.architecture), workgroup, worksize)

    advect_particles_event = advect_particles_kernel!(model.particles, model.grid, Δt, model.velocities,
                                                      dependencies=Event(device(model.architecture)))

    wait(device(model.architecture), advect_particles_event)

    return nothing
end

end # module