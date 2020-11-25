module LagrangianParticleTracking

export LagrangianParticles, advect_particles!

using Adapt
using KernelAbstractions
using StaticArrays

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

"""
    enforce_boundary_conditions(x, xₗ, xᵣ, ::Type{Bounded})

If a particle with position `x` and domain `xₗ < x < xᵣ` goes through the edge of the domain
along a `Bounded` dimension, put them back at the wall.
"""
@inline enforce_boundary_conditions(x, xₗ, xᵣ, ::Type{Bounded}) = clamp(x, xₗ, xᵣ)

"""
    enforce_boundary_conditions(x, xₗ, xᵣ, ::Type{Periodic})

If a particle with position `x` and domain `xₗ < x < xᵣ` goes through the edge of the domain
along a `Periodic` dimension, put them on the other side.
"""
@inline function enforce_boundary_conditions(x, xₗ, xᵣ, ::Type{Periodic})
    x > xᵣ && return xₗ + (x - xᵣ)
    x < xₗ && return xᵣ - (xₗ - x)
    return x
end

@kernel function _advect_particles!(particles, grid::RegularCartesianGrid{FT, TX, TY, TZ}, Δt, velocities) where {FT, TX, TY, TZ}
    p = @index(Global)

    # Advect particles using forward Euler.
    @inbounds particles.x[p] += interpolate(velocities.u, Face, Cell, Cell, grid, particles.x[p], particles.y[p], particles.z[p]) * Δt
    @inbounds particles.y[p] += interpolate(velocities.v, Cell, Face, Cell, grid, particles.x[p], particles.y[p], particles.z[p]) * Δt
    @inbounds particles.z[p] += interpolate(velocities.w, Cell, Cell, Face, grid, particles.x[p], particles.y[p], particles.z[p]) * Δt

    # Enforce boundary conditions for particles.
    @inbounds particles.x[p] = enforce_boundary_conditions(particles.x[p], grid.xF[1], grid.xF[grid.Nx], TX)
    @inbounds particles.y[p] = enforce_boundary_conditions(particles.y[p], grid.yF[1], grid.yF[grid.Ny], TY)
    @inbounds particles.z[p] = enforce_boundary_conditions(particles.z[p], grid.zF[1], grid.zF[grid.Nz], TZ)
end

advect_particles!(::Nothing, model, Δt) = nothing

function advect_particles!(particles, model, Δt)
    workgroup = min(length(particles), MAX_THREADS_PER_BLOCK)
    worksize = length(particles)
    advect_particles_kernel! = _advect_particles!(device(model.architecture), workgroup, worksize)

    advect_particles_event = advect_particles_kernel!(particles, model.grid, Δt, datatuple(model.velocities),
                                                      dependencies=Event(device(model.architecture)))

    wait(device(model.architecture), advect_particles_event)

    return nothing
end

advect_particles!(model, Δt) = advect_particles!(model.particles, model, Δt)

Adapt.adapt_structure(to, particles::LagrangianParticles) =
    LagrangianParticles(Adapt.adapt(to, particles.x), Adapt.adapt(to, particles.y), Adapt.adapt(to, particles.z))

end # module
