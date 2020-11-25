module LagrangianParticleTracking

export LagrangianParticles, advect_particles!

using Adapt
using KernelAbstractions
using StructArrays

using Oceananigans.Grids
using Oceananigans.Architectures: device
using Oceananigans.Fields: interpolate, datatuple
using Oceananigans.Utils: MAX_THREADS_PER_BLOCK

import Base: size, length

abstract type AbstractParticle end

struct Particle{T} <: AbstractParticle
    x :: T
    y :: T
    z :: T
end

struct LagrangianParticles{P,R}
      particles :: P
    restitution :: R
end

function LagrangianParticles(; x, y, z, restitution=1.0)
    size(x) == size(y) == size(z) ||
        error("x, y, z must all have the same size!")

    (ndims(x) == 1 && ndims(y) == 1 && ndims(z) == 1) ||
        error("x, y, z must have dimension 1 but ndims=($(ndims(x)), $(ndims(y)), $(ndims(z)))")

    particles = StructArray{Particle}((x, y, z))

    return LagrangianParticles(particles, restitution)
end

size(lagrangian_particles::LagrangianParticles) = size(lagrangian_particles.particles)
length(lagrangian_particles::LagrangianParticles) = length(lagrangian_particles.particles)

include("advect_particles.jl")

end # module
